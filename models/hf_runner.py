import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import ModelRunner

try:
    from peft import PeftModel
except Exception:  # noqa: BLE001
    PeftModel = None

logger = logging.getLogger(__name__)


class HFRunner(ModelRunner):
    def __init__(
        self,
        base_model: str,
        lora_source: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        self.base_model = base_model
        self.lora_source = lora_source
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading model %s on %s", base_model, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if lora_source and PeftModel:
            logger.info("Applying LoRA adapter from %s", lora_source)
            self.model = PeftModel.from_pretrained(self.model, lora_source)
        else:
            if lora_source and not PeftModel:
                logger.warning("PEFT not available; skipping adapter load")
        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def _build_prompt(self, messages: List[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "do_sample": (temperature if temperature is not None else self.temperature) > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded.strip()
