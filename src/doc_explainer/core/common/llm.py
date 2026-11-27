"""LLM wrapper interface shared by every core package that talks to a model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
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
