from abc import ABC, abstractmethod
from typing import Optional

from ...base.interfaces import RecommendationStrategy
from ...models.dataclasses import Resource
from src.core.common.enums import ResourceType


class BaseRecommendationStrategy(RecommendationStrategy, ABC):
    """Base class for recommendation strategies"""
    
    def __init__(self, recommender):
        self.recommender = recommender
    
    @abstractmethod
    def recommend(self, concept: str, level: str) -> Resource:
        """Recommend a resource"""
        pass
    
    @abstractmethod
    def get_resource_type(self) -> ResourceType:
        """Get the resource type this strategy handles"""
        pass
    
    def _create_fallback(self, concept: str, level: str) -> Resource:
        """Create a fallback recommendation"""
        return self.recommender._fallback_recommendation(
            concept, level, self.get_resource_type()
        )