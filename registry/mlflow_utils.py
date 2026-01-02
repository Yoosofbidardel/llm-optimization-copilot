import os
from typing import Optional

try:
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None


def get_client():
    if mlflow is None:
        raise RuntimeError("mlflow is not installed.")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.MlflowClient()


def log_params_and_metrics(params: dict, metrics: dict, artifacts_path: Optional[str] = None):
    if mlflow is None:
        print("MLflow not available; skipping logging.")
        return
    mlflow.start_run()
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if artifacts_path:
        mlflow.log_artifacts(artifacts_path)
    mlflow.end_run()
