from typing import Optional, Dict, Any

from ...agent import Agent

from ..engines.adaptive_explainer import AdaptiveExplainer
from ..recommenders.resource_recommender import ResourceRecommender
from ..config import ExplanationEngineConfig


class ExplanationEngineFactory:
    """Factory for creating explanation engines"""
    
    @classmethod
    def create_adaptive_explainer(cls, 
                                  agent: Agent,
                                  recommender: Optional[ResourceRecommender] = None,
                                  config: Optional[ExplanationEngineConfig] = None) -> AdaptiveExplainer:
        """Create an adaptive explainer"""
        if config is None:
            config = ExplanationEngineConfig()
        
        if recommender is None:
            recommender = ResourceRecommender()
        
        return AdaptiveExplainer(
            agent=agent,
            recommender=recommender,
            default_level=config.default_level
        )
    
    @classmethod
    def create_default(cls, agent: Agent) -> AdaptiveExplainer:
        """Create default explanation engine"""
        return cls.create_adaptive_explainer(agent)
    
    @classmethod
    def create_with_custom_recommender(cls, agent: Agent, 
                                       recommender: ResourceRecommender) -> AdaptiveExplainer:
        """Create engine with custom recommender"""
        return AdaptiveExplainer(
            agent=agent,
            recommender=recommender
        )
    
    @classmethod
    def from_config(cls, agent: Agent, config_dict: Dict[str, Any]) -> AdaptiveExplainer:
        """Create engine from configuration dictionary"""
        config = ExplanationEngineConfig.from_dict(config_dict)
        return cls.create_adaptive_explainer(agent, config=config)