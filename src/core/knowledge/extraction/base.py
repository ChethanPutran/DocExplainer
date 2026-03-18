from abc import ABC, abstractmethod
from typing import List, Any
from src.core.knowledge.models.concept import Concept

class BaseConceptExtractionStrategy(ABC):
    """Base class for concept extraction strategies"""
    
    @abstractmethod
    def extract(self, text: str) -> List[str]:
        """Extract concept candidates from text"""
        pass

class BaseRelationshipExtractionStrategy(ABC):
    """Base class for relationship extraction strategies"""
    
    @abstractmethod
    def extract(self, concepts: List[Concept], text: str, context: str) -> List[Any]:
        """Extract relationships between concepts"""
        pass

class BaseScoringStrategy(ABC):
    """Base class for concept scoring strategies"""
    
    @abstractmethod
    def score(self, concept: Concept, context: str) -> float:
        """Score a concept based on context"""
        pass

class BaseFilterStrategy(ABC):
    """Base class for concept filtering strategies"""
    
    @abstractmethod
    def filter(self, concepts: List[Concept], **kwargs) -> List[Concept]:
        """Filter concepts based on criteria"""
        pass