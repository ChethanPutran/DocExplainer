from enum import Enum


class ExplanationDepth(Enum):
    """Depth of explanation"""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"


class ExplanationStyleEnum(Enum):
    """Style/level of explanation"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ResourceType(Enum):
    """Type of learning resource"""
    VIDEO = "video"
    ARTICLE = "article"
    EXERCISE = "exercise"


class QueryType(Enum):
    """Type of user query"""
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    EXAMPLE = "example"
    SIMPLIFICATION = "simplification"
    REPETITION = "repetition"
    GENERAL = "general"