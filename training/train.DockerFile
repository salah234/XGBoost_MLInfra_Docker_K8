## Docker Notes
## In Docker you work from ROOT if need dependencies in ROOT and do down the branch 
## After that copy in any files that's needed or utilized in this DockerFile.
## docker build - builds image (-f if for specific Docker file and -t to use a name for a tag related to that specific image)
## COPY . .  - copies everything in the current directory (example right now in training) to destination (docker working dir in docker container (/app))

# Build from ROOT (xgboost_ml_infra_docker_k8)
# Docker context starts at root, so reference files relative to root

# In requirements.txt -> sometimes NEED TO add the version of the library
# xgboost==2.0.3
# scikit-learn==1.4.2

FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ../feature-pipeline ./feature-pipeline
WORKDIR /app/training
CMD ["python3", "train.py"]



