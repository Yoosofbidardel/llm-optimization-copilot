import logging
import os
from typing import Optional

from api.schemas import ProblemSpec
from models.hf_runner import HFRunner
from parser import guards, prompts

logger = logging.getLogger(__name__)


class ParserService:
    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        lora_source = os.environ.get("LORA_SOURCE") or model_cfg.get("lora_source")
        if lora_source is None and os.environ.get("MLFLOW_REGISTRY_URI"):
            from registry.resolve_model import resolve_lora_path

            resolved = resolve_lora_path(
                os.environ.get("REGISTRY_MODEL_NAME", "optim-copilot-parser"),
                os.environ.get("REGISTRY_STAGE", "Staging"),
            )
            lora_source = resolved
        self.model_runner = HFRunner(
            base_model=model_cfg.get("base_model"),
            lora_source=lora_source,
            temperature=model_cfg.get("temperature", 0.0),
            max_new_tokens=model_cfg.get("max_new_tokens", 512),
        )
        self.max_retries = 3

    def _build_messages(self, prompt: str) -> list:
        return [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "assistant", "content": prompts.EXEMPLAR_JSON},
            {"role": "user", "content": prompt},
        ]

    def generate_raw(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        messages = self._build_messages(prompt)
        return self.model_runner.generate(messages, temperature, max_new_tokens)

    def parse_problem_spec(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> ProblemSpec:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            raw = self.generate_raw(prompt, temperature, max_new_tokens)
            try:
                return guards.validate_problem_spec(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Validation failed on attempt %s: %s", attempt, exc)
        raise ValueError(f"Failed to parse ProblemSpec after {self.max_retries} attempts: {last_error}")
