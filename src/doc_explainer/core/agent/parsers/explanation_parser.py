from typing import Any, Type
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from .base import BaseParser
from ..models.schemas import ExplanationPydantic


class ExplanationParser(BaseParser):
    """Parser for explanation outputs"""
    
    def __init__(self, pydantic_object: Type[BaseModel] = ExplanationPydantic):
        super().__init__()
        self.pydantic_object = pydantic_object
    
    def parse(self, text: str) -> ExplanationPydantic:
        """Parse text into ExplanationPydantic"""
        return super().parse(text)
    
    def parse_response(self, output: str) -> ExplanationPydantic:
        """Parse text into ExplanationPydantic"""
        return self.parse(output)


# Singleton instance
explanation_output_parser = ExplanationParser()