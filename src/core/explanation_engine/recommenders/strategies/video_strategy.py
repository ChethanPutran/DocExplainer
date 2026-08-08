from .base import BaseRecommendationStrategy
from ...models.dataclasses import Resource
from src.core.common.enums import ResourceType


class VideoRecommendationStrategy(BaseRecommendationStrategy):
    """Strategy for video recommendations"""
    
    def recommend(self, concept: str, level: str) -> Resource:
        """Recommend a video resource"""
        return self.recommender.recommend_videos(concept, level)
    
    def get_resource_type(self) -> ResourceType:
        """Get resource type"""
        return ResourceType.VIDEO