from typing import Optional, cast
from langchain_core.language_models.base import BaseLanguageModel
from ..base import BaseLLM


class TestWrapper(BaseLLM):
    """Test wrapper for unit testing"""
    
    def _create_model(self, **kwargs) -> BaseLanguageModel:
        """Create a dummy model that echoes input"""
        class DummyModel:
            def __init__(self, temperature: float = 0.7):
                self.temperature = temperature
            
            def generate(self, prompt: str) -> str:
                return f"Echo: {prompt} (temp={self.temperature})"
        
        return cast(BaseLanguageModel, DummyModel(temperature=self.temperature))
    