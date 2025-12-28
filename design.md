## Design <br>
Main Goal: <br>
Through using a XGBoost ML Model, how can we containerize this model after training and inference/prediction and later on scale to multiple containers.

Need Multiple DockerFiles since each phase is different compared to rest and having one big dockerfile would to smaller builds and poor scaling

ML Task:
Using UCIML computer hardware dataset to predict published relative performance of a given CPU.




Directory:
    - requirements.txt
    - training
        - train.py (save in parquet file since column major format and its more efficient in large datasets (in DMLS Book))
        - train.DockerFile
    - Inference/Prediction
        - inference.py
        - inference.DockerFile
    - monitor
        - monitor.py
    -  k8s (Kubernetes Clusters) <- After Docker.
        - train.yaml
        - inference.yaml
        - monitoring.yaml
    - featurePipeLine
        - feature_pipeline.py -> transforms feature into clean ML-ready data
            - Transfer clean df to parquet file for Train and Inference files
    - Workflows
        - Retrain and Orchestration (Use)




