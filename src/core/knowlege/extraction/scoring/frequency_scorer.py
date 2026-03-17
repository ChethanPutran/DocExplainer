import math
from .base import BaseScoringStrategy
from src.core.knowledge.models.concept import Concept

class FrequencyScoringStrategy(BaseScoringStrategy):
    """Score concepts based on frequency in text"""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def score(self, concept: Concept, context: str) -> float:
        """Calculate frequency score (log scaled)"""
        context_lower = context.lower()
        raw_freq = 0
        
        for alias in concept.aliases:
            alias_lower = alias.lower()
            start = 0
            while True:
                pos = context_lower.find(alias_lower, start)
                if pos == -1:
                    break
                raw_freq += 1
                start = pos + len(alias_lower)
        
        if raw_freq == 0:
            return 0.0
            
        return self.weight * math.log1p(raw_freq)