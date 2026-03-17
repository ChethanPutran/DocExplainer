from typing import Optional, Dict, Any
from ..base import BaseLLM
from ..wrappers.gemini_wrapper import GeminiWrapper
from ..wrappers.openai_wrapper import OpenAIWrapper


class LLMFactory:
    """Factory for creating LLM wrappers"""
    
    _providers = {
        'gemini': GeminiWrapper,
        'openai': OpenAIWrapper,
        'google': GeminiWrapper,
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
        return wrapper_class(**kwargs)
    
    @classmethod
    def create_default(cls, **kwargs) -> BaseLLM:
        """Create default LLM (Gemini)"""
        return cls.create('gemini', **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())