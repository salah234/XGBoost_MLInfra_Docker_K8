from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import boto3
import os
from dotenv import load_dotenv
from uuid import uuid4
import json

from datetime import datetime, timezone

load_dotenv()
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
S3_INFERENCE_BUCKET = os.getenv("S3_BUCKET_INFERENCE")
s3 = boto3.client('s3',
                  aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=AWS_REGION
)

S3_PREFIX='ml_inference'

app = FastAPI()
BASE_DIR = Path("/tmp/training")
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = BASE_DIR / "xgb_model.json"
FEATURES_FILE = BASE_DIR / "feature_names.joblib"

# Only download files if missing (want to do in tmp) 
# Need training files (Features scaled on and XGBoost ML Model)
if not MODEL_FILE.exists():
    s3.download_file(S3_INFERENCE_BUCKET, "training/xgb_model.json", str(MODEL_FILE))
if not FEATURES_FILE.exists():
    s3.download_file(S3_INFERENCE_BUCKET, "training/feature_names.joblib", str(FEATURES_FILE))


# # Load model and features
# # use UUID in the s3 object payload since its dynamic (user changes feature based on specific prediction needed)

model = XGBRegressor()
model.load_model(MODEL_FILE)
feature_names = joblib.load(FEATURES_FILE)["feature_names"]



def save_to_s3_object(data: dict) -> str:
    object_key = f"{S3_PREFIX}/{uuid4()}.json"
    s3.put_object(Bucket=S3_INFERENCE_BUCKET,
                  Key=object_key,
                  Body=json.dumps(data),
                  ContentType='application/json')
    return object_key



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
    s3_payload = {
        "inputs": data,
        "prediction": float(prediction),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print(s3_payload)
    save_to_s3_object(s3_payload)

    return f"""
        <h1>Prediction Result</h1>
        <p>Predicted CPU performance: {prediction:.2f}</p>
        <a href="/">Back</a>
    """
