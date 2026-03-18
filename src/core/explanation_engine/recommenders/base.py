from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from ..base.interfaces import ResourceRecommenderInterface, RecommendationStrategy
from ..models.dataclasses import Resource
from ..models.enums import ResourceType, ExplanationLevel

logger = logging.getLogger(__name__)


class BaseResourceRecommender(ResourceRecommenderInterface, ABC):
    """Base class for resource recommenders"""
    
    def __init__(self):
        self.strategies: List[RecommendationStrategy] = []
        self._register_default_strategies()
    
    @abstractmethod
    def _register_default_strategies(self):
        """Register default recommendation strategies"""
        pass
    
    def register_strategy(self, strategy: RecommendationStrategy):
        """Register a recommendation strategy"""
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.__class__.__name__}")
    
    def get_strategy_for_type(self, resource_type: ResourceType) -> Optional[RecommendationStrategy]:
        """Get strategy for a specific resource type"""
        for strategy in self.strategies:
            if strategy.get_resource_type() == resource_type:
                return strategy
        return None
    
    def recommend_videos(self, concept: str, level: str) -> Resource:
        """Recommend videos using video strategy"""
        strategy = self.get_strategy_for_type(ResourceType.VIDEO)
        if strategy:
            return strategy.recommend(concept, level)
        return self._fallback_recommendation(concept, level, ResourceType.VIDEO)
    
    def recommend_articles(self, concept: str, level: str) -> Resource:
        """Recommend articles using article strategy"""
        strategy = self.get_strategy_for_type(ResourceType.ARTICLE)
        if strategy:
            return strategy.recommend(concept, level)
        return self._fallback_recommendation(concept, level, ResourceType.ARTICLE)
    
    def recommend_exercises(self, concept: str, level: str) -> Resource:
        """Recommend exercises using exercise strategy"""
        strategy = self.get_strategy_for_type(ResourceType.EXERCISE)
        if strategy:
            return strategy.recommend(concept, level)
        return self._fallback_recommendation(concept, level, ResourceType.EXERCISE)
    
    def recommend_all(self, concept: str, level: str) -> List[Resource]:
        """Recommend all resource types"""
        resources = []
        
        for strategy in self.strategies:
            try:
                resource = strategy.recommend(concept, level)
                resources.append(resource)
            except Exception as e:
                logger.error(f"Strategy {strategy.__class__.__name__} failed: {e}")
                # Add fallback
                resources.append(self._fallback_recommendation(
                    concept, level, strategy.get_resource_type()
                ))
        
        return resources
    
    def _fallback_recommendation(self, concept: str, level: str, 
                                 resource_type: ResourceType) -> Resource:
        """Create a fallback recommendation when strategies fail"""
        platform_map = {
            ResourceType.VIDEO: "youtube",
            ResourceType.ARTICLE: "scholar",
            ResourceType.EXERCISE: "google"
        }
        
        platform = platform_map.get(resource_type, "google")
        url = Resource.create_search_link(concept, level, platform)
        
        type_names = {
            ResourceType.VIDEO: "Video",
            ResourceType.ARTICLE: "Article",
            ResourceType.EXERCISE: "Exercise"
        }
        
        return Resource(
            title=f"{type_names.get(resource_type, 'Resource')}: {concept} ({level})",
            url=url,
            type=resource_type,
            description=f"Search results for {concept} at {level} level",
            difficulty=ExplanationLevel(level) if isinstance(level, str) else level
        )