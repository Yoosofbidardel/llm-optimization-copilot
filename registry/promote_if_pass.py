import json
import os
from pathlib import Path

from registry.mlflow_utils import get_client


def main():
    report_path = Path("evals/eval_report.json")
    if not report_path.exists():
        raise FileNotFoundError("eval_report.json not found. Run evals/run_evals.py first.")
    report = json.loads(report_path.read_text())
    metrics = report.get("metrics", {})
    if not metrics.get("success"):
        raise SystemExit("Metrics did not meet threshold; refusing to promote.")

    model_name = os.environ.get("REGISTRY_MODEL_NAME", "optim-copilot-parser")
    stage = os.environ.get("REGISTRY_STAGE", "Staging")

    client = get_client()
    latest = client.get_latest_versions(model_name, stages=["None"])
    if not latest:
        raise SystemExit("No model versions available to promote.")
    version = latest[0].version
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    print(f"Promoted {model_name} version {version} to stage {stage}")


if __name__ == "__main__":
    main()
