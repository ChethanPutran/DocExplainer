from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...core.common.dataclasses import ExplanationStyle

@dataclass
class AgentConfig:
    """Configuration for agent"""
    
    # LLM Configuration
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    temperature: float = 1.0
    max_retries: int = 3
    llm_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "gemini-3.5-flash-lite",
        "max_tokens": None,
        "timeout": None,
        "requests_per_minute": 4,
        "rate_limit_retries": 2
    })
    
    # Default styles
    explanation_style: ExplanationStyle = field(
    default_factory=ExplanationStyle.get_default_style
    )
    
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

        if 'llm_model' in config_dict:
            config.llm_model = config_dict['llm_model']
            config.llm_kwargs['model_name'] = config.llm_model
        
        if 'temperature' in config_dict:
            config.temperature = config_dict['temperature']
        
        if 'max_retries' in config_dict:
            config.max_retries = config_dict['max_retries']
        
        if 'llm_kwargs' in config_dict:
            config.llm_kwargs.update(config_dict['llm_kwargs'])
        
        if 'explanation_style' in config_dict:
            style_val = config_dict['explanation_style']
            if isinstance(style_val, dict):
                config.explanation_style = ExplanationStyle.from_dict(style_val)
            else:
                config.explanation_style = ExplanationStyle.get_default_style()
                

        if 'tone' in config_dict:
            config.tone = config_dict['tone']
        
        if 'math_level' in config_dict:
            config.math_level = config_dict['math_level']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'llm_kwargs': self.llm_kwargs,
            'explanation_style': self.explanation_style,
            'tone': self.tone,
            'math_level': self.math_level
        }
