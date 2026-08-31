from typing import Optional
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from ..base import BaseLLM
from langchain_openrouter import ChatOpenRouter
import logging

logger = logging.getLogger(__name__)

class OpenRouterWrapper(BaseLLM):
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning: Optional[dict] = None,
        **kwargs,
    ):
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self.base_url = base_url
        self.api_key = api_key
        self.reasoning = reasoning

        super().__init__(model_name, temperature, **kwargs)

    def _create_model(self, **kwargs):

        logger.info(f"Creating OpenRouter model: {self.model_name} with temperature={self.temperature}, \
                    max_tokens={self.max_tokens}, timeout={self.timeout}, max_retries={self.max_retries},\
                      base_url={self.base_url}, reasoning={self.reasoning}")


        return ChatOpenRouter(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            reasoning={"effort": "high"},
            **kwargs,
        )