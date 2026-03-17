from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BKTConfig:
    """Bayesian Knowledge Tracing configuration"""
    initial_knowledge: float = 0.1
    learning_rate: float = 0.3
    guess_probability: float = 0.2
    slip_probability: float = 0.1
    confidence_threshold: float = 0.7
    memory_decay: float = 0.85

@dataclass
class UserModuleConfig:
    """User module configuration"""
    bkt: BKTConfig = field(default_factory=BKTConfig)
    storage_path: str = "data/user_data/"
    interactions_path: str = "data/interactions/"
    auto_save: bool = True
    enable_text_inference: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UserModuleConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'bkt' in config_dict:
            bkt_dict = config_dict['bkt']
            config.bkt.initial_knowledge = bkt_dict.get('initial_knowledge', 0.1)
            config.bkt.learning_rate = bkt_dict.get('learning_rate', 0.3)
            config.bkt.guess_probability = bkt_dict.get('guess_probability', 0.2)
            config.bkt.slip_probability = bkt_dict.get('slip_probability', 0.1)
            config.bkt.confidence_threshold = bkt_dict.get('confidence_threshold', 0.7)
            config.bkt.memory_decay = bkt_dict.get('memory_decay', 0.85)
        
        config.storage_path = config_dict.get('storage_path', 'data/user_data/')
        config.interactions_path = config_dict.get('interactions_path', 'data/interactions/')
        config.auto_save = config_dict.get('auto_save', True)
        config.enable_text_inference = config_dict.get('enable_text_inference', True)
        
        return config