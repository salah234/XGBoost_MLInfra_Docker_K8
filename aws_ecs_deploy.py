import boto3 
import os 
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("AWS_SECRET_KEY")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")


ecs = boto3.client('ecs', region_name=AWS_REGION)


