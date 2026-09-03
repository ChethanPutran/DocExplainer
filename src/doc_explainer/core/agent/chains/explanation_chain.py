from typing import Optional
from langchain_core.runnables import Runnable

from .base import BaseChain
from ..llm.base import BaseLLM
from ..parsers.base import BaseParser
from ..prompts.templates import explain_prompt
from ..models.schemas import ExplanationPydantic


class ExplanationChain(BaseChain):
    """Chain for generating explanations"""
    
    def __init__(self, llm: BaseLLM, parser: Optional[BaseParser] = None):
        super().__init__(llm, parser)
        self.llm.set_prompt_template(explain_prompt, json_output=True)
    
    def _build_chain(self) -> Runnable:
        """Build explanation chain"""
        chain = self.llm.chain
        if chain is None:
            raise ValueError("LLM chain is not initialized")
        return chain
    
    def run(self, **kwargs) -> ExplanationPydantic:
        """Run explanation chain"""
        tone = kwargs.pop("tone", "encouraging and academic")
        complexity = kwargs.pop("complexity", "intermediate")
        math_level = kwargs.pop("math_level", "descriptive")
        kwargs.setdefault("question", kwargs.get("selected_text", ""))

        result = super().run(
            tone=tone,
            complexity=complexity,
            math_level=math_level,
            **kwargs,
        )
        # Normalize result to ExplanationPydantic
        return self.normalize_explanation_result(result)