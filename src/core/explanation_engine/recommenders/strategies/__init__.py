from .base import BaseRecommendationStrategy
from .video_strategy import VideoRecommendationStrategy
from .article_strategy import ArticleRecommendationStrategy
from .exercise_strategy import ExerciseRecommendationStrategy

__all__ = [
    'BaseRecommendationStrategy',
    'VideoRecommendationStrategy',
    'ArticleRecommendationStrategy',
    'ExerciseRecommendationStrategy'
]