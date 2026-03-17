from typing import Optional
from langchain_core.runnables import RunnableSequence

from .base import BaseChain
from ..llm.base import BaseLLM
from ..parsers.base import BaseParser
from ..prompts.templates import summarize_prompt


class SummarizationChain(BaseChain):
    """Chain for generating summaries"""
    
    def __init__(self, llm: BaseLLM, parser: Optional[BaseParser] = None):
        super().__init__(llm, parser)
        self.llm.set_prompt_template(summarize_prompt, json_output=True)
    
    def _build_chain(self) -> RunnableSequence:
        """Build summarization chain"""
        return self.llm.chain
    
    def run(self, selected_text: str, context_summary: str,
            known_concepts: str, complexity: str = "intermediate",
            structure: str = "bullet points",
            length: str = "concise") -> Any:
        """Run summarization chain"""
        return super().run(
            selected_text=selected_text,
            context_summary=context_summary,
            known_concepts=known_concepts,
            complexity=complexity,
            structure=structure,
            length=length
        )