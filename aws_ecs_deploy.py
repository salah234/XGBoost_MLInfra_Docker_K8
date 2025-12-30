import boto3  # AWS SDK for Python
import os 
from dotenv import load_dotenv
import base64

# Using ECS on AWS Fargate to run inference and training 
# Workflow:
    # Amazon ECR to create repositories to store the docker images in.
    # Amazon ECS to define how to run the containers
    # AWS Fargate to run the containers in a serverless manner
    # Need Auth Token for ECR

load_dotenv()

SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")



ecr = boto3.client('ecr', region_name=AWS_REGION)
repo_stages = ['xgb-train', 'inference'] # Match it with docker image tages
for repo in repo_stages:
    try:
        ecr.create_repository(repositoryName=repo)
        print(f"Created ECR Repository for {repo}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
          print(f"Repository {repo} already exists")

    

auth = ecr.get_authorization_token()
token = auth['authorizationData'][0]['authorizationToken']
user, password = base64.b64decode(token).decode().split(':')
registry = auth['authorizationData'][0]['proxyEndpoint']





