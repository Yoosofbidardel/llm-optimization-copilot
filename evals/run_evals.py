import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import httpx
import yaml

try:
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None


def load_cases() -> List[Dict]:
    cases_dir = Path("evals/cases")
    return [json.loads(path.read_text()) for path in cases_dir.glob("*.json")]


def evaluate_case(client: httpx.Client, case: Dict):
    start = time.perf_counter()
    parse_resp = client.post("/parse", json={"prompt": case["prompt"]}, timeout=120)
    parse_latency = time.perf_counter() - start
    parse_ok = parse_resp.status_code == 200
    parse_json = parse_resp.json() if parse_ok else {}

    solve_ok = False
    solve_json = {}
    if parse_ok:
        solve_resp = client.post("/solve", json={"spec": parse_json}, timeout=120)
        solve_ok = solve_resp.status_code == 200
        solve_json = solve_resp.json() if solve_ok else {}

    return {
        "parse_ok": parse_ok,
        "parse_latency": parse_latency,
        "solve_ok": solve_ok,
        "parse_json": parse_json,
        "solve_json": solve_json,
    }


def compute_metrics(results: List[Dict], thresholds: Dict) -> Dict:
    latencies = [r["parse_latency"] for r in results if r["parse_ok"]]
    parse_rate = sum(1 for r in results if r["parse_ok"]) / len(results)
    solve_rate = sum(1 for r in results if r["solve_ok"]) / len(results)
    p50 = statistics.median(latencies) if latencies else None
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else p50
    success = True
    if parse_rate < thresholds["parse_success_threshold"]:
        success = False
    if solve_rate < thresholds["solve_success_threshold"]:
        success = False
    if p95 and p95 * 1000 > thresholds["latency_p95_threshold_ms"]:
        success = False
    return {
        "parse_json_rate": parse_rate,
        "e2e_success_rate": solve_rate,
        "latency_p50_ms": p50 * 1000 if p50 else None,
        "latency_p95_ms": p95 * 1000 if p95 else None,
        "success": success,
    }


def main():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    thresholds = config.get("evals", {})
    api_host = os.environ.get("API_HOST", config["api"]["host"])
    api_port = int(os.environ.get("API_PORT", config["api"]["port"]))
    base_url = f"http://{api_host}:{api_port}"

    client = httpx.Client(base_url=base_url)
    cases = load_cases()
    results = [evaluate_case(client, case) for case in cases]
    metrics = compute_metrics(results, thresholds)

    report = {"cases": results, "metrics": metrics}
    report_path = Path("evals/eval_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote eval report to {report_path}")
    print(json.dumps(metrics, indent=2))

    if mlflow and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.start_run(run_name="optim-copilot-evals")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(report_path))
        mlflow.end_run()

    if not metrics["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
