from typing import Optional
from langchain_core.runnables import RunnableSequence

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
    
    def _build_chain(self) -> RunnableSequence:
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

        result = super().run(
            tone=tone,
            complexity=complexity,
            math_level=math_level,
            **kwargs,
        )
        # Normalize result to ExplanationPydantic
        if isinstance(result, ExplanationPydantic):
            return result
        if isinstance(result, dict):
            return ExplanationPydantic(**result)

        # If result is not a dict or the expected pydantic model, raise to satisfy typing
        raise TypeError(f"Unexpected result type {type(result)!r}, expected dict or ExplanationPydantic")