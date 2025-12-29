from ucimlrepo import fetch_ucirepo 
import os
import boto3 # To use ECS for deployment (Need to upload the data to an s3 bucket so that later can use it with the dockerfile to run ECS)
from dotenv import load_dotenv

load_dotenv()
  
# fetch dataset 
RAW_DATASET_ID = 29
OUTPUT_DIR = "feature-pipeline"
OUTPUT_FILE = "computer_hardware_clean.parquet"
BUCKET = os.getenv('S3_BUCKET')
s3_client = boto3.client('s3')



os.makedirs(OUTPUT_DIR, exist_ok=True)

def upload_to_s3(local_path, bucket, key):
    s3_client.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} -> s3://{bucket}/{key}")

def main():
    computer_hardware = fetch_ucirepo(id=29) 

    # Don't Standardize since XGBoost is an ensemble-tree based and split off of thresholds instead of distances.
    df = computer_hardware.data.features 
    df_clean = df.drop(columns=['VendorName', 'ModelName'], errors='ignore')
    df_clean = df_clean.fillna(df_clean.mean()) # Mean Imputation

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_clean.to_parquet(output_path, index=False)
    print(f"Clean and sent Parquet File to: {output_path}")

    upload_to_s3(output_path, BUCKET, f"{OUTPUT_DIR}/{OUTPUT_FILE}")


if __name__ == "__main__":
    main()

    
    





