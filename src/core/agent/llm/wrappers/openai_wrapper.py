from typing import Optional
from langchain_openai import ChatOpenAI
from ..base import BaseLLM


class OpenAIWrapper(BaseLLM):
    """Wrapper for OpenAI models"""
    
    def __init__(self,
                 model_name: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None,
                 max_retries: int = 2,
                 **kwargs):
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        super().__init__(model_name, temperature, **kwargs)
    
    def _create_model(self, **kwargs) -> ChatOpenAI:
        """Create OpenAI model"""
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            **kwargs
        )