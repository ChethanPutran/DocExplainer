from .base import BaseRecommendationStrategy
from ...models.dataclasses import Resource
from ....common.enums import ResourceType


class ArticleRecommendationStrategy(BaseRecommendationStrategy):
    """Strategy for article recommendations"""
    
    def recommend(self, concept: str, level: str) -> Resource:
        """Recommend an article resource"""
        return self.recommender.recommend_articles(concept, level)
    
    def get_resource_type(self) -> ResourceType:
        """Get resource type"""
        return ResourceType.ARTICLE