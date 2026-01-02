from abc import ABC, abstractmethod
from typing import List, Optional


class ModelRunner(ABC):
    @abstractmethod
    def generate(self, messages: List[dict], temperature: Optional[float] = None, max_new_tokens: Optional[int] = None) -> str:
        raise NotImplementedError
