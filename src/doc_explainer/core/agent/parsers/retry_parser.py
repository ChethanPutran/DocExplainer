from typing import Any, Optional
from langchain_core.language_models import BaseLanguageModel

from .base import BaseParser


class RetryParser(BaseParser):
    """Parser with retry capability"""
    
    def __init__(self, parser: BaseParser, llm: BaseLanguageModel, max_retries: int = 3):
        super().__init__()
        self.parser = parser
        self.llm = llm
        self.max_retries = max_retries
    
    def parse(self, text: str) -> Any:
        """Parse with retry on failure"""
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                return self.parser.parse(text)
            except Exception as e:
                last_error = e
                attempts += 1
        
        raise last_error or RuntimeError(f"Failed to parse after {self.max_retries} attempts")
    
    def parse_with_prompt(self, text: str, prompt: str) -> Any:
        """Parse with access to original prompt for retry"""
        return self.parse(text)
    
    def get_format_instructions(self) -> str:
        """Get format instructions from underlying parser"""
        return self.parser.get_format_instructions()