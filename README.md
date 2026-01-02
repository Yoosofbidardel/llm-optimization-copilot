# Optimisation Copilot

An end-to-end demo that turns natural language optimisation requests into validated JSON specs and solves them with OR-Tools. The project is organised for reliability (schema + guardrails), performance (lazy model load), and MLOps (MLflow-friendly training, registry, and evaluation flows). Everything runs locally on Windows (PowerShell) and Linux, with CPU fallbacks when GPUs are unavailable. Run commands from the repository root.

## Repository Structure

- `api/` — FastAPI service that orchestrates parsing and solving.
- `parser/` — NL → JSON compiler with strict prompts and guardrails.
- `models/` — Hugging Face runner with optional LoRA adapter loading.
- `solver/` — Deterministic OR-Tools assignment solver and explanation helpers.
- `ui/` — Streamlit UI that calls the API.
- `training/` — Synthetic data generation, QLoRA fine-tuning, quick parser eval.
- `evals/` — Gold cases and evaluation harness with MLflow logging.
- `registry/` — MLflow registry helpers (resolve, promote).
- `configs/` — YAML configuration (API host/port, model knobs, thresholds).
- `docker/` — Docker Compose for API + UI + MLflow.

## Architecture & Design Decisions

### Separation of Concerns
- **Parser (compiler):** Converts NL to a typed `ProblemSpec` JSON; it never solves.
- **Solver:** Deterministic OR-Tools assignment with capacities and cost minimisation.
- **API:** Thin orchestration layer; lazy-loads the model on the first parse call; `/healthz` is always instant and never loads the model.
- **UI:** Streamlit client only; no embedded model logic—always calls the API.
- **Training/Evals:** Offline, reproducible scripts that do not run in the API process.
- **Registry:** MLflow-friendly stubs for promotion and adapter resolution.

### Reliability
- **Strict schema:** `ProblemSpec` Pydantic model enforces required keys and types.
- **Guardrails:** Strip code fences, extract the first JSON object, JSON-load, and validate.
- **Retries:** Parser retries up to 3 times if validation fails.
- **Determinism:** Temperature defaults to 0.0; retries can override via request.
- **Debuggability:** `/parse_raw` returns the untouched model output; `/parse` returns validated JSON; `/healthz` is fast and model-free.

### Performance
- **Lazy model load:** `HFRunner` loads on first parse invocation, never during import or `/healthz`.
- **Selective decoding:** Only decodes newly generated assistant tokens (no prompt transcript).
- **Chat templates:** Uses `tokenizer.apply_chat_template` when available, with fallback.
- **Configurable generation:** Temperature and `max_new_tokens` are configurable per-request and via YAML.

### MLOps
- **MLflow tracking:** Training and eval scripts optionally log params/metrics/artifacts when `MLFLOW_TRACKING_URI` is set.
- **Adapter handling:** LoRA adapters are saved as artifacts; `registry/resolve_model.py` can pull the latest staged adapter from MLflow.
- **Promotion workflow:** `registry/promote_if_pass.py` promotes the latest model version to a target stage when evals succeed.

### Evaluation
- **Ground truth:** Gold JSON specs in `evals/cases/*.json` capture the expected structure and solution. They represent the canonical interpretation of the NL prompt.
- **Metrics:** `evals/run_evals.py` computes parse_json_rate, e2e_success_rate (solver OPTIMAL/FEASIBLE), p50/p95 parse latency, and writes `eval_report.json`.
- **Exit codes:** Evaluations exit with code 1 if thresholds are not met (configurable in `configs/default.yaml`), making them CI-friendly.

## Configuration

`configs/default.yaml` controls API host/port, model name, LoRA source, temperature, max tokens, solver flags, evaluation thresholds, and registry defaults. Override via environment variables:

- `API_HOST`, `API_PORT` — API binding (port 8010 by default).
- `BASE_MODEL`, `LORA_SOURCE` — Model identifiers; `LORA_SOURCE` can be resolved from MLflow registry when `MLFLOW_REGISTRY_URI` is set.
- `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI` — Enable logging and registry resolution.
- `REGISTRY_MODEL_NAME`, `REGISTRY_STAGE` — Target model and stage for adapter resolution/promotion.
- `API_URL` — UI target API (defaults to `http://127.0.0.1:8010`).

## FastAPI Endpoints (port 8010)
- `GET /healthz` — Instant heartbeat; no model load.
- `POST /parse_raw` — Returns raw model output for debugging.
- `POST /parse` — Returns validated `ProblemSpec`.
- `POST /solve` — Solves a `ProblemSpec` with OR-Tools and returns assignments/explanation.
- `GET /` — Redirects to `/docs`.

## Parser Prompting & Validation
- **System prompt:** Demands strict JSON with `tasks`, `resources`, `must_assign_all_tasks`.
- **Exemplar:** Includes a minified JSON example to anchor structure.
- **Guardrails:** Remove code fences, extract the first `{...}` block, JSON-load, then Pydantic-validate.
- **Retries:** Up to 3 attempts with the same prompt; request-level temperature/max tokens can adjust behaviour.

## Solver Semantics
- **Model:** Assignment with capacities per resource.
- **Objective:** Minimise total cost.
- **Constraints:** If `must_assign_all_tasks` is true, every task must be assigned exactly once; otherwise tasks may be skipped. Resource capacities enforced.
- **Output:** Status (OPTIMAL/FEASIBLE/etc), total cost, per-task assignments, and a human explanation.

## Training
- **Synthetic data:** `training/synth_data.py` writes `data/train.jsonl` and `data/val.jsonl` with NL ↔ JSON pairs.
- **LoRA fine-tuning:** `training/finetune_lora.py` supports QLoRA on CUDA via BitsAndBytes (guarded) with CPU fallback. It masks prompt tokens with -100 for proper loss, uses `remove_unused_columns=False`, saves the adapter to `artifacts/lora`, and logs to MLflow when configured.
- **Parser sanity check:** `training/eval_parser.py` runs a quick parse using the configured model.

### Adapter Versioning & Registry
- **Tracking:** Training logs params/metrics and uploads adapters when MLflow is configured.
- **Promotion:** After passing evals, run `registry/promote_if_pass.py` to move the latest version to a stage (e.g., Staging/Production).
- **Resolution:** `registry/resolve_model.py` fetches the staged adapter path for loading in the API. If unavailable, the base model is used.

## Evaluation Harness
- **Cases:** Gold specs live in `evals/cases/`.
- **Runner:** `evals/run_evals.py` calls `/parse` and `/solve`, measures latency, and writes `eval_report.json`. Exit code is non-zero when thresholds fail.
- **Metrics:** `parse_json_rate`, `e2e_success_rate`, `latency_p50_ms`, `latency_p95_ms`.
- **Ground truth meaning:** The gold JSON is the canonical representation of the NL prompt; success is measured by schema validity and solver feasibility against that spec.

## Deployment Flow
1. Generate data → fine-tune (`training/finetune_lora.py`) → logs to MLflow → adapter artifact saved.
2. Run evals (`evals/run_evals.py`); inspect `eval_report.json`.
3. Promote if passing (`registry/promote_if_pass.py`) to desired stage.
4. API loads the staged adapter via `LORA_SOURCE` or registry resolution; UI connects via `API_URL`.

## Windows & PyCharm Instructions

### PowerShell Setup (CPU-safe)
```powershell
Set-Location llm-optimization-copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Run API (PowerShell)
```powershell
Set-Location llm-optimization-copilot
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

### Streamlit UI (PowerShell)
```powershell
Set-Location llm-optimization-copilot
$env:API_URL="http://127.0.0.1:8010"
streamlit run ui/app.py --server.port 8502
```

### PyCharm Run Configs
- **API:** Module name `uvicorn`, parameters `api.main:app --host 127.0.0.1 --port 8010`, working directory = repo root, environment variables `API_HOST=127.0.0.1;API_PORT=8010`.
- **UI:** Module name `streamlit`, parameters `run ui/app.py --server.port 8502`, environment `API_URL=http://127.0.0.1:8010`.

### Common Windows Fixes
- **Symlink warnings (HF Hub):** Set `HF_HUB_DISABLE_SYMLINKS=1`.
- **Port already in use:** `netstat -a -n -o | findstr 8010` then `taskkill /PID <pid> /F`.
- **localhost vs 127.0.0.1:** Use `127.0.0.1` explicitly for API/UI.
- **GPU not detected:** Check `python - <<<'import torch; print(torch.cuda.is_available())'`; if `False`, scripts automatically fall back to CPU.
- **Slow first run:** Model downloads and warmup may take time; subsequent runs are faster.

## Acceptance Checklist (expected behaviour)
1. `pip install -r requirements.txt`
2. `uvicorn api.main:app --host 127.0.0.1 --port 8010`
3. `GET http://127.0.0.1:8010/healthz` returns `{"status":"ok"}` instantly.
4. `POST /parse` with a sample prompt returns a valid `ProblemSpec`.
5. `POST /solve` returns `OPTIMAL` with assignments respecting capacity and cost.
6. `streamlit run ui/app.py --server.port 8502` (with `API_URL=127.0.0.1:8010`) shows the UI and works end-to-end.
7. `python training/synth_data.py` writes `data/train.jsonl` and `data/val.jsonl`.
8. `python training/finetune_lora.py` runs; uses CPU if CUDA unavailable.
9. `python evals/run_evals.py` produces `eval_report.json` and exits 0 when thresholds pass.

## Troubleshooting
- **Model download fails:** Check internet connectivity or set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads.
- **BitsAndBytes on Windows:** It is optional and skipped; training falls back to standard precision.
- **Validation errors in /parse:** Use `/parse_raw` to inspect model output; reduce temperature or provide clearer costs/capacities.
- **OR-Tools wheel issues:** A broad version range (`>=9.10,<9.12`) is used to align with available Windows wheels; upgrade pip if install fails.
- **Slow parsing:** Lower `max_new_tokens` or choose a smaller `BASE_MODEL`.

## How to Run (Quickstart)
```bash
cd llm-optimization-copilot
python -m venv .venv
source .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8010
# In another terminal:
API_URL=http://127.0.0.1:8010 streamlit run ui/app.py --server.port 8502
```
