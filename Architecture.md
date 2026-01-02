# Optimisation Copilot Architecture

This repo turns natural language optimisation requests into validated JSON specs and deterministic solutions. The stack is split into clear runtime and offline paths for reliability, performance, and MLOps.

## Runtime Flow (API + UI)

```
Users ──(browser)──▶ Streamlit UI
                      │ calls HTTP (API_URL)
                      ▼
                FastAPI service (api/)
                      │ lazy-init ParserService on first parse
                      ▼
               ParserService (parser/)
                      │ build system prompt + exemplar
                      │ call model runner
                      ▼
             HFRunner (models/) ──▶ Base model + optional LoRA
                      │ decode assistant tokens only
                      ▼
        Guardrails (parser/guards.py)
                      │ strip fences, extract first {...}, json-load
                      ▼
              Pydantic schemas (api/schemas.py)
                      │ ensure ProblemSpec validity
                      ▼
                 Solver (solver/) ──▶ OR-Tools (SCIP)
                      │ enforce capacities, must_assign_all_tasks flag
                      ▼
          Solution + explanation text returned to UI
```

### Key design decisions
- **Separation of concerns:** UI is a thin client; API orchestrates; parser compiles NL→JSON; solver is deterministic and model-free.
- **Reliability:** Strict Pydantic schema, guardrails before validation, retries (3) on parse failures, fast `/healthz` with no model load.
- **Performance:** Lazy model load on first parse, selective decoding of new tokens, temperature 0.0 by default, chat template when available.
- **Configurability:** `configs/default.yaml` + env vars control host/port, model IDs, generation knobs, registry targets.
- **Safety & determinism:** OR-Tools solver minimises cost with explicit constraints; optional must-assign toggle; low-temperature generation.

## Offline Pipelines (Training, Eval, Registry)

```
Synthetic data (training/synth_data.py)
          │
          ▼
 LoRA fine-tune (training/finetune_lora.py) ──▶ artifacts/lora + MLflow logging (optional)
          │
          ▼
 Quick parser eval (training/eval_parser.py)

Gold cases (evals/cases/*.json)
          │
          ▼
 Eval harness (evals/run_evals.py) ──▶ parse_json_rate, e2e_success_rate, latency metrics → eval_report.json
          │
          ▼
 Registry helpers (registry/resolve_model.py, promote_if_pass.py) ──▶ pull/push staged adapters via MLflow
```

### Design decisions
- **MLOps friendly:** Optional MLflow tracking/registry integration; adapters saved as artifacts; promotion helper after passing evals.
- **Portability:** CPU fallbacks if CUDA unavailable; broad OR-Tools range for wheel availability; works on Windows/Linux.
- **Reproducibility:** Deterministic eval exit codes for CI; YAML-driven defaults; data and artifacts written under repo tree.

## Deployment & Ops

- **Docker Compose (`docker/`):** Brings up API, UI, and MLflow (optional) with sensible ports; UI points to API service.
- **Config surface:** `API_HOST`/`API_PORT` for binding; `BASE_MODEL`/`LORA_SOURCE` for model selection; registry envs for adapter resolution.
- **Health:** `/healthz` is instant (no model load); `/parse_raw` exposes raw model output for debugging; `/parse` returns validated spec; `/solve` returns solver result + explanation.

