from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator"""
    
    # Pipeline configuration
    default_user_id: str = "user_123"
    concepts_per_para: int = 10
    
    # LLM configuration
    llm_provider: str = "gemini"
    temperature: float = 1.0
    
    # Storage paths
    persist_directory: str = "db/vector_dbs"
    knowledge_store_path: str = "db/knowledge_graph.gpickle"
    memory_path: str = "data/memory/user_memory.json"
    
    # Explanation style
    explanation_style: str = "intermediate"
    
    # Feature flags
    enable_knowledge_graph: bool = True
    enable_memory: bool = True
    enable_session_tracking: bool = True
    
    # Additional kwargs
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OrchestratorConfig':
        """Create config from dictionary"""
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key == 'llm_kwargs' and isinstance(value, dict):
                    config.llm_kwargs.update(value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'default_user_id': self.default_user_id,
            'concepts_per_para': self.concepts_per_para,
            'llm_provider': self.llm_provider,
            'temperature': self.temperature,
            'persist_directory': self.persist_directory,
            'knowledge_store_path': self.knowledge_store_path,
            'memory_path': self.memory_path,
            'explanation_style': self.explanation_style,
            'enable_knowledge_graph': self.enable_knowledge_graph,
            'enable_memory': self.enable_memory,
            'enable_session_tracking': self.enable_session_tracking,
            'llm_kwargs': self.llm_kwargs
        }