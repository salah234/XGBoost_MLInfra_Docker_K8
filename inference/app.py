from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
from xgboost import XGBRegressor
import joblib
from pathlib import Path

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parents[1]

# Load model and features
model = XGBRegressor()
model.load_model(BASE_DIR / "training" / "xgb_model.json")
feature_names = joblib.load(BASE_DIR / "training" / "feature_names.joblib")["feature_names"]

@app.get("/", response_class=HTMLResponse)
def home():
    return """
        <h1>CPU Performance Prediction</h1>
        <form action="/generate-results" method="post">
            MYCT: <input type="number" name="MYCT"><br>
            MMIN: <input type="number" name="MMIN"><br>
            MMAX: <input type="number" name="MMAX"><br>
            CACH: <input type="number" name="CACH"><br>
            CHMIN: <input type="number" name="CHMIN"><br>
            CHMAX: <input type="number" name="CHMAX"><br>
            ERP: <input type="number" name="ERP"><br>
            <input type="submit" value="Predict">
        </form>
    """

@app.post("/generate-results", response_class=HTMLResponse)
def predict(
    MYCT: int = Form(...),
    MMIN: int = Form(...),
    MMAX: int = Form(...),
    CACH: int = Form(...),
    CHMIN: int = Form(...),
    CHMAX: int = Form(...),
    ERP: int = Form(...)
):
    # Prepare input for model
    data = {
        "MYCT": MYCT,
        "MMIN": MMIN,
        "MMAX": MMAX,
        "CACH": CACH,
        "CHMIN": CHMIN,
        "CHMAX": CHMAX,
        "ERP": ERP
    }
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df)[0]

    return f"""
        <h1>Prediction Result</h1>
        <p>Predicted CPU performance: {prediction:.2f}</p>
        <a href="/">Back</a>
    """
