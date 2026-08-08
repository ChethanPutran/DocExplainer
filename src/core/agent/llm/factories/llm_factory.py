from ..base import BaseLLM
from ..wrappers.gemini_wrapper import GeminiWrapper
from ..wrappers.local_wrapper import LocalWrapper
from ..wrappers.ollama_wrapper import OllamaWrapper
from ..wrappers.openai_wrapper import OpenAIWrapper
from ..wrappers.open_router import OpenRouterWrapper


class LLMFactory:
    """Factory for creating LLM wrappers"""
    
    _providers = {
        'gemini': GeminiWrapper,
        'local': LocalWrapper,
        'ollama': OllamaWrapper,
        'openai': OpenAIWrapper,
        'openrouter': OpenRouterWrapper
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
        return wrapper_class(**kwargs) # Create instance of the wrapper class with provided kwargs
    
    @classmethod
    def create_default(cls, **kwargs) -> BaseLLM:
        """Create default LLM (Gemini)"""
        return cls.create('gemini', **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())
