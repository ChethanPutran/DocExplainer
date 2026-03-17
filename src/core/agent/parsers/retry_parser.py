from typing import Any, Optional
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.language_models import BaseLanguageModel

from .base import BaseParser


class RetryParser(BaseParser):
    """Parser with retry capability"""
    
    def __init__(self, parser: BaseParser, llm: BaseLanguageModel, max_retries: int = 3):
        super().__init__()
        self.parser = parser
        self.retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=parser,
            llm=llm,
            max_retries=max_retries
        )
    
    def parse(self, text: str) -> Any:
        """Parse with retry on failure"""
        return self.retry_parser.parse(text)
    
    def parse_with_prompt(self, text: str, prompt: str) -> Any:
        """Parse with access to original prompt for retry"""
        return self.retry_parser.parse_with_prompt(text, prompt_value=prompt)
    
    def get_format_instructions(self) -> str:
        """Get format instructions from underlying parser"""
        return self.parser.get_format_instructions()