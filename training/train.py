import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from pathlib import Path
import boto3 
import os 

s3 = boto3.client("s3")
BUCKET = os.getenv("S3_BUCKET")

INPUT_KEY = "feature-pipeline/computer_hardware_clean.parquet"
MODEL_KEY = "training/xgb_model.json"
FEATURES_KEY = "training/feature_names.joblib"


def download_from_s3(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded s3://{bucket}/{key} -> {local_path}")

def upload_to_s3_bucket(bucket, key, local_path):
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} -> s3://{bucket}/{key}")

def main():
    BASE_DIR = Path("/tmp") # VERY IMPORTANT FOR DOING AWS ECS since /tmp crucial for temporary data storage (So make a path for our files to go in)
    INPUT_FILE = BASE_DIR / "feature-pipeline" / "computer_hardware_clean.parquet"
    MODEL_OUTPUT = BASE_DIR / "training" / "xgb_model.json"
    FEATURES_OUTPUT = BASE_DIR / "training" / "feature_names.joblib"

    INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    download_from_s3(BUCKET, INPUT_KEY, str(INPUT_FILE))
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
    upload_to_s3_bucket(BUCKET, MODEL_KEY, str(MODEL_OUTPUT))
    upload_to_s3_bucket(BUCKET, FEATURES_KEY, str(FEATURES_OUTPUT))

    y_pred = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.3f}")


if __name__ == "__main__":
    main()
