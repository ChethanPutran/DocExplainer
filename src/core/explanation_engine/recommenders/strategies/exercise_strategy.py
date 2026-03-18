from .base import BaseRecommendationStrategy
from ...models.dataclasses import Resource
from ...models.enums import ResourceType


class ExerciseRecommendationStrategy(BaseRecommendationStrategy):
    """Strategy for exercise recommendations"""
    
    def recommend(self, concept: str, level: str) -> Resource:
        """Recommend an exercise resource"""
        return self.recommender.recommend_exercises(concept, level)
    
    def get_resource_type(self) -> ResourceType:
        """Get resource type"""
        return ResourceType.EXERCISE