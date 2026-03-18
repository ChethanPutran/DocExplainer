from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models.dataclasses import Resource
from ..models.enums import ResourceType, ExplanationLevel


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
    def set_explanation_level(self, level: ExplanationLevel):
        """Set explanation level"""
        pass


class ResourceRecommenderInterface(ABC):
    """Interface for resource recommenders"""
    
    @abstractmethod
    def recommend_videos(self, concept: str, level: str) -> 'Resource':
        """Recommend videos for concept"""
        pass
    
    @abstractmethod
    def recommend_articles(self, concept: str, level: str) -> 'Resource':
        """Recommend articles for concept"""
        pass
    
    @abstractmethod
    def recommend_exercises(self, concept: str, level: str) -> 'Resource':
        """Recommend exercises for concept"""
        pass
    
    @abstractmethod
    def recommend_all(self, concept: str, level: str) -> List['Resource']:
        """Recommend all resource types"""
        pass


class RecommendationStrategy(ABC):
    """Strategy for recommending specific resource types"""
    
    @abstractmethod
    def recommend(self, concept: str, level: str) -> 'Resource':
        """Recommend a resource"""
        pass
    
    @abstractmethod
    def get_resource_type(self) -> ResourceType:
        """Get the resource type this strategy handles"""
        pass