from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ConceptExtractionConfig:
    """Configuration for concept extraction"""
    concepts_per_para: int = 10
    min_concept_length: int = 2
    max_concept_length: int = 50
    enable_llm_refinement: bool = True
    enable_statistical_relations: bool = True
    similarity_threshold: float = 0.85
    scoring_strategies: List[str] = field(default_factory=lambda: [
        "frequency", "position", "length", "definition"
    ])

@dataclass
class GraphConfig:
    """Configuration for knowledge graph"""
    max_graph_size: int = 10000
    validate_cycles: bool = True
    enable_visualization: bool = True
    persist_graph: bool = True
    graph_storage_path: str = "data/knowledge_graphs/"

@dataclass
class KnowledgeModuleConfig:
    """Main configuration for knowledge module"""
    extraction: ConceptExtractionConfig = field(default_factory=ConceptExtractionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        config = cls()
        if 'extraction' in config_dict:
            for key, value in config_dict['extraction'].items():
                if hasattr(config.extraction, key):
                    setattr(config.extraction, key, value)
        if 'graph' in config_dict:
            for key, value in config_dict['graph'].items():
                if hasattr(config.graph, key):
                    setattr(config.graph, key, value)
        return config