
import boto3 
import os 
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("AWS_SECRET_KEY")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

bucket_name = 'ml-train-bucket-23'

s3 = boto3.client('s3', region_name=AWS_REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)

print(AWS_REGION)

if AWS_REGION == "us-east-1":
    s3.create_bucket(Bucket=bucket_name)
else:
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
    )




print(f"Bucket {bucket_name} created successfully")



