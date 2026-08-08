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

class ExplanationDepth(Enum):
    """Depth of explanation"""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
