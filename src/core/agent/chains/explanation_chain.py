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
        return self.llm.chain
    
    def run(self, selected_text: str, context_summary: str, 
            known_concepts: str, unknown_concepts: str,
            tone: str = "encouraging and academic",
            complexity: str = "intermediate",
            math_level: str = "descriptive") -> ExplanationPydantic:
        """Run explanation chain"""
        result = super().run(
            selected_text=selected_text,
            context_summary=context_summary,
            known_concepts=known_concepts,
            unknown_concepts=unknown_concepts,
            tone=tone,
            complexity=complexity,
            math_level=math_level
        )
        
        if isinstance(result, dict) and self.parser:
            return ExplanationPydantic(**result)
        return result