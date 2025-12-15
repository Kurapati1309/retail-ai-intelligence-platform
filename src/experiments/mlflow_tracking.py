import os
import mlflow

def setup_mlflow(experiment_name="RAIIP"):
    uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
