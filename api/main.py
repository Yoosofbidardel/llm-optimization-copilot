import json
import os
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.schemas import ProblemSpec, Solution
from parser.service import ParserService
from solver.service import solve_problem

CONFIG_PATH = Path(os.environ.get("OPTIM_COPILOT_CONFIG", "configs/default.yaml"))


class ParseRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None


class SolveRequest(BaseModel):
    spec: ProblemSpec


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


app = FastAPI(title="Optimisation Copilot API", version="0.1.0")
_parser_service: Optional[ParserService] = None
_config_cache = load_config()


def get_parser_service() -> ParserService:
    global _parser_service
    if _parser_service is None:
        _parser_service = ParserService(_config_cache)
    return _parser_service


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/parse_raw")
async def parse_raw(request: ParseRequest):
    parser = get_parser_service()
    raw = parser.generate_raw(
        prompt=request.prompt,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens,
    )
    return {"raw": raw}


@app.post("/parse", response_model=ProblemSpec)
async def parse(request: ParseRequest):
    parser = get_parser_service()
    try:
        spec = parser.parse_problem_spec(
            prompt=request.prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )
        return spec
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/solve", response_model=Solution)
async def solve(request: SolveRequest):
    try:
        solution = solve_problem(request.spec)
        return solution
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main():
    host = os.environ.get("API_HOST", _config_cache.get("api", {}).get("host", "127.0.0.1"))
    port = int(os.environ.get("API_PORT", _config_cache.get("api", {}).get("port", 8010)))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
