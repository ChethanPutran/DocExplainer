from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel


class LLMInterface(ABC):
    """Interface for LLM wrappers"""
    
    @abstractmethod
    def generate(self, inputs: Dict[str, Any]) -> Any:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def set_prompt_template(self, template: PromptTemplate, json_output: bool = False):
        """Set prompt template"""
        pass
    
    @abstractmethod
    def get_model(self) -> BaseLanguageModel:
        """Get underlying model"""
        pass


class ParserInterface(ABC):
    """Interface for output parsers"""
    pydantic_model = None

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse LLM output"""
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get format instructions for prompts"""
        pass


class ChainInterface(ABC):
    """Interface for processing chains"""
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the chain"""
        pass
    
    @abstractmethod
    def with_retry(self, max_retries: int = 3) -> 'ChainInterface':
        """Add retry capability"""
        pass