import logging
import threading
from typing import Dict

from ..base import BaseLLM
from ..wrappers.airouter import AIRouterWrapper
from ..wrappers.gemini_wrapper import GeminiWrapper
from ..wrappers.local_wrapper import LocalWrapper
from ..wrappers.ollama_wrapper import OllamaWrapper
from ..wrappers.openai_wrapper import OpenAIWrapper
from ..wrappers.open_router import OpenRouterWrapper


logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM wrappers.

    Wrappers asked for with identical settings share ONE underlying model client and one rate limiter
    (see ``BaseLLM.fork``); each caller still gets its own wrapper for its own prompt and chain.
    """

    _shared: Dict[str, BaseLLM] = {}
    _lock = threading.Lock()
    
    _providers = {
        'gemini': GeminiWrapper,
        'local': LocalWrapper,
        'ollama': OllamaWrapper,
        'openai': OpenAIWrapper,
        'openrouter': OpenRouterWrapper,
        'airouter': AIRouterWrapper,
    }
    
    @classmethod
    def register_provider(cls, name: str, wrapper_class):
        """Register a new provider"""
        cls._providers[name.lower()] = wrapper_class
    
    @classmethod
    def create(cls, provider: str = 'gemini', **kwargs) -> BaseLLM:
        """Create an LLM wrapper"""
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._providers.keys())}")
        
        wrapper_class = cls._providers[provider]
        instance_name = kwargs.get("instance_name", "default")
        # Same provider + same settings (the role's name aside) = same client. Kept in memory only.
        key = repr((provider, sorted((k, v) for k, v in kwargs.items() if k != "instance_name")))
        with cls._lock:
            base = cls._shared.get(key)
            if base is None:
                base = cls._shared[key] = wrapper_class(**kwargs)  # the one real initialisation
            else:
                logger.info("Sharing the existing %s client for '%s'", provider, instance_name)
        return base.fork(instance_name)

    @classmethod
    def clear_shared(cls) -> None:
        """Forget the shared clients (tests, or after the configuration changed)."""
        with cls._lock:
            cls._shared.clear()
    
    @classmethod
    def create_default(cls, **kwargs) -> BaseLLM:
        """Create default LLM (Gemini)"""
        return cls.create('gemini', **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())
