from abc import ABC, abstractmethod

from typing import List
from src.core.knowledge.models.concept import Concept

class BaseScoringStrategy(ABC):
    """Base class for concept scoring strategies"""
    
    @abstractmethod
    def score(self, concept: Concept, context: List[str]) -> float:
        """Score a concept based on context"""
        pass