from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .models.enums import ExplanationStyleEnum


@dataclass
class AgentConfig:
    """Configuration for agent"""
    
    # LLM Configuration
    llm_provider: str = "gemini"
    temperature: float = 1.0
    max_retries: int = 3
    llm_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": None,
        "timeout": None
    })
    
    # Default styles
    default_style: ExplanationStyleEnum = ExplanationStyleEnum.INTERMEDIATE
    tone: str = "encouraging and academic"
    math_level: str = "descriptive"
    
    # Chain configuration
    enable_retry: bool = True
    cache_responses: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AgentConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'llm_provider' in config_dict:
            config.llm_provider = config_dict['llm_provider']
        
        if 'temperature' in config_dict:
            config.temperature = config_dict['temperature']
        
        if 'max_retries' in config_dict:
            config.max_retries = config_dict['max_retries']
        
        if 'llm_kwargs' in config_dict:
            config.llm_kwargs.update(config_dict['llm_kwargs'])
        
        if 'default_style' in config_dict:
            style_val = config_dict['default_style']
            if isinstance(style_val, str):
                config.default_style = ExplanationStyleEnum(style_val)
        
        if 'tone' in config_dict:
            config.tone = config_dict['tone']
        
        if 'math_level' in config_dict:
            config.math_level = config_dict['math_level']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'llm_provider': self.llm_provider,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'llm_kwargs': self.llm_kwargs,
            'default_style': self.default_style.value,
            'tone': self.tone,
            'math_level': self.math_level
        }