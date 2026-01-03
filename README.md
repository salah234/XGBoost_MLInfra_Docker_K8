# XGBoost ML Infrastructure with Docker & AWS ECS - Fargate

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Docker](https://img.shields.io/badge/docker-latest-blue)]()
[![Kubernetes](https://img.shields.io/badge/kubernetes-latest-blue)]()

## Project Overview

This repository provides a **complete ML infrastructure** for training, deploying, and serving **XGBoost models** with:

- Python for ML pipelines and inference  
- Docker for containerization  
- AWS ECS and Fargate for cloud deployment  
- S3 for model and data storage
- CloudWatch for Task initialization and Logging

It includes training scripts, feature engineering pipelines, inference services, and deployment configurations.

## Next Steps:
Extend to be able to handle Kubernetes clusters & pods.
---
```
## Repository Structure
├── training/ # Training scripts & model files <br>
│ ├── train.py
│ ├── xgb_model.joblib
│ ├── xgb_model.json
│ └── train.DockerFile
├── inference/ # Inference API & Docker container
│ ├── app.py
│ ├── inference.DockerFile
│ ├── s3_utils.py
│ └── pycache/
├── feature-pipeline/ # Feature engineering scripts & data
│ ├── feature-pipeline.py
│ └── computer_hardware_clean.parquet
├── .env # Environment variables (ignored)
├── requirements.txt # Python dependencies
├── aws_ecs_deploy.py # AWS ECS deployment script
├── k8_ml.yaml # Kubernetes deployment config
└── design.md # Project design notes
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/username/XGBoost_ML_Infra_Docker_K8.git
cd XGBoost_ML_Infra_Docker_K8

