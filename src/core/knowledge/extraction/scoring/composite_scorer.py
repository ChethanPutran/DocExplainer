from typing import List
from .base import BaseScoringStrategy
from src.core.knowledge.models.concept import Concept

class CompositeScoringStrategy(BaseScoringStrategy):
    """Combine multiple scoring strategies"""
    
    def __init__(self, strategies: List[BaseScoringStrategy]):
        self.strategies = strategies
    
    def score(self, concept: Concept, context: str) -> float:
        """Calculate combined score from all strategies"""
        score = 1.0  # Base score
        
        for strategy in self.strategies:
            score *= strategy.score(concept, context)
        
        return score