from .base import BaseScoringStrategy
from ....knowledge.models.concept import Concept

class LengthScoringStrategy(BaseScoringStrategy):
    """Score concepts based on name length (prefer 2-3 words)"""
    
    def score(self, concept: Concept, context: str) -> float:
        """Calculate length multiplier"""
        num_words = len(concept.name.split())
        
        if num_words == 2:
            return 1.4
        elif num_words == 3:
            return 1.7
        elif num_words >= 4:
            return 0.7
        else:  # 1 word
            return 1.0