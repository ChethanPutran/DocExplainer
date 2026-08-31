import math
from .base import BaseScoringStrategy
from ....knowledge.models.concept import Concept

class PositionScoringStrategy(BaseScoringStrategy):
    """Score concepts based on their first occurrence position"""
    
    def __init__(self, decay_factor: float = 2.5):
        self.decay_factor = decay_factor
    
    def score(self, concept: Concept, context: str) -> float:
        """Calculate position score (earlier = more important)"""
        context_lower = context.lower()
        text_len = len(context_lower)
        
        if text_len == 0:
            return 0.0
        
        # Find first occurrence across all aliases
        first_pos = text_len
        for alias in concept.aliases:
            pos = context_lower.find(alias.lower())
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        if first_pos == text_len:
            return 0.0
        
        relative_pos = first_pos / text_len
        return math.exp(-self.decay_factor * relative_pos)