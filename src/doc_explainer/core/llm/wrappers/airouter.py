"""AIRouter (https://www.airouter.in): an OpenAI-compatible gateway.

Models are ``<provider>/<model>`` slugs. The model, endpoint and key are always given by the caller
(``config/defaults/llm_config.yaml`` holds the defaults); list the models your key can use with ``GET <base_url>/models``.
"""
import logging
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..base import BaseLLM

logger = logging.getLogger(__name__)



class AIRouterWrapper(BaseLLM):
    """LLM wrapper for AIRouter, using the OpenAI client pointed at the gateway."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
        base_url: Optional[str] = None,    # required: the caller (config) supplies the endpoint
        api_key: Optional[str] = None,
        **kwargs,
    ):
        key = api_key or os.getenv("AIROUTER_API_KEY")
        if not key:
            raise ValueError("AIRouter needs an API key: set AIROUTER_API_KEY (for example in .env).")
        # Private names: BaseLLM overwrites self.max_tokens from kwargs, which would drop ours.
        self._api_key = SecretStr(key)
        if not base_url:
            raise ValueError("AIRouter needs an endpoint: set providers.airouter.base_url in llm_config.yaml.")
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        super().__init__(model_name, temperature, **kwargs)

    def _create_model(self, **kwargs) -> ChatOpenAI:
        logger.info("Creating AIRouter model: %s (temperature=%s, max_tokens=%s, timeout=%s, "
                    "max_retries=%s, base_url=%s)", self.model_name, self.temperature,
                    self._max_tokens, self._timeout, self._max_retries, self._base_url)
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
            base_url=self._base_url,
            api_key=self._api_key,
            **{k: v for k, v in kwargs.items() if k != "max_tokens"},
        )
