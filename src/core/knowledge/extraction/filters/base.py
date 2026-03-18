from abc import ABC, abstractmethod
from typing import List
from src.core.knowledge.models.concept import Concept

class BaseFilterStrategy(ABC):
    """Base class for concept filtering strategies"""
    
    @abstractmethod
    def filter(self, concepts: List[Concept], **kwargs) -> List[Concept]:
        """Filter concepts based on criteria"""
        pass