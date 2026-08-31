from abc import ABC, abstractmethod
from typing import List

from ...models.concept import Concept
from ...models.relationship import ConceptRelationship

class BaseConceptExtractionStrategy(ABC):
    """Base class for concept extraction strategies"""
    
    @abstractmethod
    def extract(self, text: str) -> List[str]:
        """Extract concept candidates from text"""
        pass
    
class BaseRelationshipExtractionStrategy(ABC):
    """Base class for relationship extraction strategies"""
    
    @abstractmethod
    def extract(self, concepts: List[Concept], text: str, context: str) -> List[ConceptRelationship]:
        """Extract relationships from concepts and text"""
        pass