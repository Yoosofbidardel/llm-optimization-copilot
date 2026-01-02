import json
import re
from typing import Any

from api.schemas import ProblemSpec

JSON_BLOCK_REGEX = re.compile(r"\{.*\}", re.DOTALL)


def strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    return text.replace("```json", "").replace("```", "").strip()


def extract_json_block(text: str) -> str:
    cleaned = strip_code_fences(text)
    match = JSON_BLOCK_REGEX.search(cleaned)
    if not match:
        raise ValueError("No JSON object found in model output")
    return match.group(0)


def validate_problem_spec(text: str) -> ProblemSpec:
    block = extract_json_block(text)
    try:
        data: Any = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode JSON: {exc}") from exc
    try:
        return ProblemSpec.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to validate ProblemSpec: {exc}") from exc
