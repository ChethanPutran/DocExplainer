from enum import Enum


class ResourceType(Enum):
    """Types of learning resources"""
    VIDEO = "video"
    ARTICLE = "article"
    EXERCISE = "exercise"
    COURSE = "course"
    DOCUMENTATION = "documentation"


class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class RecommendationStrategy(str, Enum):
    """Strategies for recommendation"""
    SIMPLE_SEARCH = "simple_search"
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_BASED = "knowledge_based"
    POPULARITY_BASED = "popularity_based"