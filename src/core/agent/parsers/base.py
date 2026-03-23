from abc import abstractmethod
from typing import Any, Type, Optional
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel

from ..base.interfaces import ParserInterface


class BaseParser(ParserInterface):
    """Base class for output parsers"""
    
    def __init__(self, pydantic_model: Optional[Type[BaseModel]] = None):
        self.pydantic_model = pydantic_model
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse LLM output"""
        pass

    def get_format_instructions(self) -> str:
        """Get format instructions for prompts"""
        if hasattr(self, 'pydantic_model') and self.pydantic_model:
            schema = self.pydantic_model.schema()
            return f"Output should be a JSON object matching this schema: {schema}"
        return "Output should be valid JSON."
    
    @property
    def _type(self) -> str:
        """Get parser type"""
        return "base_parser"