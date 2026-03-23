from typing import Optional, Any
from langchain_core.runnables import RunnableSequence

from .base import BaseChain
from ..llm.base import BaseLLM
from ..parsers.base import BaseParser
from ..prompts.templates import ask_prompt


class QAChain(BaseChain):
    """Chain for question answering"""
    
    def __init__(self, llm: BaseLLM, parser: Optional[BaseParser] = None):
        super().__init__(llm, parser)
        self.llm.set_prompt_template(ask_prompt, json_output=True)
    
    def _build_chain(self) -> RunnableSequence:
        """Build QA chain"""
        return self.llm.chain
    
    def run(self, question: str, selected_text: str,
            context_summary: str, known_concepts: str) -> Any:
        """Run QA chain"""
        return super().run(
            question=question,
            selected_text=selected_text,
            context_summary=context_summary,
            known_concepts=known_concepts
        )