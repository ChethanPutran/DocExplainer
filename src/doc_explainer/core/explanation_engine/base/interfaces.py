from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ...common.dataclasses import ExplanationStyle
from ...common.enums import ExplanationLevel, ResourceType
from ..models.dataclasses import Resource


class ExplanationEngine(ABC):
    """Interface for explanation engines"""
    
    @abstractmethod
    def summarize(self, text: str, context: Any) -> Any:
        """Generate summary for text"""
        pass
    
    @abstractmethod
    def explain(self, text: str, context: Any) -> Any:
        """Generate explanation for text"""
        pass
    
    @abstractmethod
    def ask(self, question: str, context: Any) -> Any:
        """Answer question"""
        pass
    
    @abstractmethod
    def set_explanation_style(self, style: ExplanationStyle):
        """Set explanation level"""
        pass


class ResourceRecommenderInterface(ABC):
    """Interface for resource recommenders"""
    
    @abstractmethod
    def recommend_videos(self, concept: str, level: ExplanationLevel) -> 'Resource':
        """Recommend videos for concept"""
        pass
    
    @abstractmethod
    def recommend_articles(self, concept: str, level: ExplanationLevel) -> 'Resource':
        """Recommend articles for concept"""
        pass
    
    @abstractmethod
    def recommend_exercises(self, concept: str, level: ExplanationLevel) -> 'Resource':
        """Recommend exercises for concept"""
        pass
    
    @abstractmethod
    def recommend_khan_academy(self, concept: str, level: ExplanationLevel) -> Optional['Resource']:
        """Recommend Khan Academy resources"""
        pass

    @abstractmethod
    def recommend_all(self, concept: str, level: ExplanationLevel) -> List['Resource']:
        """Recommend all resource types"""
        pass


class RecommendationStrategy(ABC):
    """Strategy for recommending specific resource types"""
    
    @abstractmethod
    def recommend(self, concept: str, level: ExplanationLevel) -> 'Resource':
        """Recommend a resource"""
        pass
    
    @abstractmethod
    def get_resource_type(self) -> ResourceType:
        """Get the resource type this strategy handles"""
        pass