import json
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate

from ...base.interfaces import LLMInterface


class LocalWrapper(LLMInterface):
    """Free local fallback wrapper for deterministic, extractive responses."""

    def __init__(self, model_name: str = "local-extractive", **_kwargs):
        self.model_name = model_name
        self.prompt_template: Optional[PromptTemplate] = None

    def set_prompt_template(self, template: PromptTemplate, json_output: bool = False):
        self.prompt_template = template
        self.json_output = json_output

    def generate(self, inputs: Dict[str, Any]) -> str:
        if "current_text" in inputs:
            return self._summarize(str(inputs.get("current_text", "")))

        if "concepts" in inputs:
            concepts = inputs.get("concepts", [])
            if isinstance(concepts, str):
                concepts = [c.strip() for c in concepts.split(",") if c.strip()]
            return json.dumps({str(concept): [str(concept)] for concept in concepts})

        if "candidates" in inputs:
            candidates = inputs.get("candidates", [])
            if isinstance(candidates, str):
                candidates = [c.strip() for c in candidates.split(",") if c.strip()]
            return json.dumps([
                {"name": str(candidate).lower(), "definition": ""}
                for candidate in candidates
            ])

        text = " ".join(str(value) for value in inputs.values())
        return self._summarize(text)

    def get_model(self):
        return None

    @staticmethod
    def _summarize(text: str, max_chars: int = 240) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        summary = sentences[0] if sentences else normalized

        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")

        return summary
