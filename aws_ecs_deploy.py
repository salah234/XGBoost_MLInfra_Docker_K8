import boto3  # AWS SDK for Python
import os 
from dotenv import load_dotenv
import base64
import json

# Using ECS on AWS Fargate to run inference and training 
# Workflow:
    # Amazon ECR to create repositories to store the docker images in.
    # Make logs for ECS Fargate Task initialization.
    # Amazon ECS to define how to run the containers
    # AWS Fargate to run the containers in a serverless manner

load_dotenv()

SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

ECS_SECRET_KEY = os.getenv("AWS_ECS_SECRET_ACCESS_KEY")
ECS_ACCESS_KEY = os.getenv("AWS_ECS_ACCESS_KEY")

ecr = boto3.client('ecr', region_name=AWS_REGION)
repo_stages = ['xgb-train', 'inference'] # Match it with docker image tages
for repo in repo_stages:
    try:
        ecr.create_repository(repositoryName=repo)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {repo} already exists")

    

auth = ecr.get_authorization_token()
token = auth['authorizationData'][0]['authorizationToken']
user, password = base64.b64decode(token).decode().split(':')
registry = auth['authorizationData'][0]['proxyEndpoint']

# Create IAM Roles for Fargate Compatibility
iam = boto3.client('iam',
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY,
                region_name=AWS_REGION)

trust_policy = {
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ecs-tasks.amazonaws.com" },
      "Action": "sts:AssumeRole",
     
    }
  ]
}


sts = boto3.client( # To Determine which IAM User/Role we're currently in
    "sts",
    aws_access_key_id=ECS_ACCESS_KEY,
    aws_secret_access_key=ECS_SECRET_KEY,
    region_name=AWS_REGION
)

identity = sts.get_caller_identity()


try:
    iam.create_role(
        RoleName='ecsTaskExecutionRole',
        AssumeRolePolicyDocument=json.dumps(trust_policy)
    )
    iam.attach_role_policy(
        RoleName='ecsTaskExecutionRole',
        PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
    )
    iam.update_assume_role_policy(
    RoleName='ecsTaskExecutionRole',
    PolicyDocument=json.dumps(trust_policy)
)
except iam.exceptions.EntityAlreadyExistsException:
    print('IAM Task Execution Resource Exists')


# One for S3 handling of ML Files
try:
    iam.create_role(
        RoleName='ecsTaskRole',
        AssumeRolePolicyDocument=json.dumps(trust_policy)
    )
    iam.attach_role_policy(
        RoleName='ecsTaskRole',
        PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
    )
except iam.exceptions.EntityAlreadyExistsException:
    print('IAM Task Role Resource Exists')

# One Cluster with 2 task definitions

ecs = boto3.client('ecs',aws_access_key_id=ECS_ACCESS_KEY,
    aws_secret_access_key=ECS_SECRET_KEY,
    region_name=AWS_REGION)
try:
    ecs.create_cluster(clusterName='ml-proj-cluster')
except ecs.exceptions.ResourceAlreadyExistsException:
    print('Resource Exists')


EXEC_ROLE = 'arn:aws:iam::133715232948:role/service-role/ecsTaskExecutionRole'
TASK_ROLE = "arn:aws:iam::133715232948:role/ecsTaskRole"

# Create AWS CloudWatch Logging for Fargate Instance in order to do task initialization before running Fargate Tasks

cloudwatch_log = boto3.client('logs',  
                                aws_access_key_id=ACCESS_KEY,
                                aws_secret_access_key=SECRET_KEY,
                                region_name=AWS_REGION)

for ecs_group in ["/ecs/inference", "/ecs/xgb-train"]: # Create Log Group sharing the same ECS settings
    try:
        cloudwatch_log.create_log_group(logGroupName=ecs_group)
        print(f"Created Log group {ecs_group}")
    except cloudwatch_log.exceptions.ResourceAlreadyExistsException:
        print(f"Log group {ecs_group} already exists")


train_task_def = {
    "family": "xgb-train-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "2048",
    "memory": "4096",
    "executionRoleArn": EXEC_ROLE,
    "taskRoleArn": TASK_ROLE,
    "containerDefinitions": [
        {
            "name": "xgb-train",
            "image": "133715232948.dkr.ecr.us-east-1.amazonaws.com/xgb-train:latest",
            "essential": True,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/xgb-train",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}

ecs.register_task_definition(**train_task_def)


inference_task_def = {
    "family": "inference-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": EXEC_ROLE,
    "taskRoleArn": TASK_ROLE,
    "containerDefinitions": [
        {
            "name": "inference",
            "image": "133715232948.dkr.ecr.us-east-1.amazonaws.com/inference:image",
            "essential": True,
            "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
            "logConfiguration": { # If add JSON, then need to hook CloudWatch Logs to ML Application
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/inference",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}

ecs.register_task_definition(**inference_task_def)


# Training: One-Off 
# Inference: Long-Running since its dynamic based on runtime from User

ecs.run_task(
    cluster='ml-proj-cluster',
    launchType='FARGATE',
    taskDefinition='xgb-train-task',
    networkConfiguration={
        'awsvpcConfiguration': {
            'subnets': ['subnet-0f1f7df5030364a6d', 'subnet-014c7df41e93a7079'], # Need to configure the subnets in AWS VPC
            'securityGroups': ['sg-067a5a6ec870228a7'],
            'assignPublicIp': 'ENABLED'
        }
    }
)

# ecs.create_service(
#     cluster='ml-proj-cluster',
#     serviceName='inference-service',
#     taskDefinition='inference-task',
#     desiredCount=1,
#     launchType='FARGATE',
#     networkConfiguration={
#         'awsvpcConfiguration': {
#             'subnets': ['subnet-0f1f7df5030364a6d', 'subnet-014c7df41e93a7079'],
#             'securityGroups': ['sg-067a5a6ec870228a7'],
#             'assignPublicIp': 'ENABLED'
#         }
#     }
# )


ecs.update_service(
    cluster='ml-proj-cluster',
    service='inference-service',
    desiredCount=1,
    taskDefinition='inference-task'
)

# See if the ECS Fargate Tasks are running
resp = ecs.list_tasks(cluster='ml-proj-cluster')
print(resp['taskArns'])

desc = ecs.describe_tasks(
    cluster='ml-proj-cluster',
    tasks=resp['taskArns']
)

for t in desc['tasks']:
    print(t['taskArn'], t['lastStatus'], t['desiredStatus'])


# For Fargate - need to build docker container for linux/amd64 since it's compatabile with AWS ECS-Fargate and optimized.