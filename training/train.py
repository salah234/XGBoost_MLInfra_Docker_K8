import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "feature-pipeline" / "computer_hardware_clean.parquet"
MODEL_OUTPUT = BASE_DIR / "training" / "xgb_model.json"
FEATURES_OUTPUT = BASE_DIR / "training" / "feature_names.joblib"

def main():
    df = pd.read_parquet(INPUT_FILE)

    X = df.drop(columns=["PRP"])
    y = df["PRP"]

    joblib.dump({"feature_names": X.columns.tolist()}, FEATURES_OUTPUT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    model.save_model(MODEL_OUTPUT)

    y_pred = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.3f}")

if __name__ == "__main__":
    main()
