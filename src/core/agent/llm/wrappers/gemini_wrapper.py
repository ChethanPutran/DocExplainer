from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from ..base import BaseLLM


class GeminiWrapper(BaseLLM):
    """Wrapper for Google Gemini models"""
    
    def __init__(self, 
                 model_name: str = "gemini-3.5-flash",
                 temperature: float = 1.0,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None,
                 max_retries: int = 2,
                 requests_per_minute: Optional[int] = 4,
                 min_request_interval_seconds: Optional[float] = None,
                 rate_limit_retries: int = 2,
                 **kwargs):
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        super().__init__(
            model_name,
            temperature,
            requests_per_minute=requests_per_minute,
            min_request_interval_seconds=min_request_interval_seconds,
            rate_limit_retries=rate_limit_retries,
            **kwargs
        )
    
    def _create_model(self, **kwargs) -> ChatGoogleGenerativeAI:
        """Create Google Gemini model"""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            **kwargs
        )
