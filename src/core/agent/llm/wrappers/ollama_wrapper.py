import json
from typing import Any, Dict, Optional

import requests
from langchain_core.prompts import PromptTemplate

from .local_wrapper import LocalWrapper


class OllamaWrapper(LocalWrapper):
    """Free local Ollama wrapper with extractive fallback."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        fallback_to_local: bool = True,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fallback_to_local = fallback_to_local

    def generate(self, inputs: Dict[str, Any]) -> str:
        prompt = self._format_prompt(inputs)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip()
        except Exception:
            if self.fallback_to_local:
                return super().generate(inputs)
            raise

    def _format_prompt(self, inputs: Dict[str, Any]) -> str:
        if self.prompt_template:
            return self.prompt_template.format(**inputs)

        return json.dumps(inputs, ensure_ascii=False)
