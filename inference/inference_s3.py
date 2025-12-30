import boto3
import os
from dotenv import load_dotenv

load_dotenv()


SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

bucket_name = 'ml-inference-bucket-23'
s3_client = boto3.client('s3', region_name=AWS_REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)

if AWS_REGION == 'us-east-1':
    s3_client.create_bucket(Bucket=bucket_name)
else:
    s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': AWS_REGION})

print(f"Created bucket for s3://{bucket_name}")