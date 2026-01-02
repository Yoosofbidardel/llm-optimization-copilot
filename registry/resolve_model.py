import os
from pathlib import Path
from typing import Optional

from registry.mlflow_utils import get_client


def resolve_lora_path(model_name: str, stage: str) -> Optional[str]:
    try:
        client = get_client()
    except Exception:
        return None
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    version = versions[0]
    download_path = client.download_artifacts(version.run_id, "lora_adapter")
    return str(Path(download_path))


def main():
    model_name = os.environ.get("REGISTRY_MODEL_NAME", "optim-copilot-parser")
    stage = os.environ.get("REGISTRY_STAGE", "Staging")
    path = resolve_lora_path(model_name, stage)
    if path:
        print(f"Resolved adapter path: {path}")
    else:
        print("No adapter found in registry; using base model only.")


if __name__ == "__main__":
    main()
