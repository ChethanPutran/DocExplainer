from .base import BaseScoringStrategy
from src.core.knowledge.models.concept import Concept

class DefinitionBonusStrategy(BaseScoringStrategy):
    """Bonus score for concepts that appear in definition patterns"""
    
    def __init__(self, bonus_multiplier: float = 2.0):
        self.bonus_multiplier = bonus_multiplier
        self.definition_patterns = [
            "{} is",
            "{} are",
            "{} refers to",
            "{}:"
        ]
    
    def score(self, concept: Concept, context: str) -> float:
        """Check if concept appears in definition patterns"""
        context_lower = context.lower()
        
        for alias in concept.aliases:
            alias_lower = alias.lower()
            for pattern in self.definition_patterns:
                if pattern.format(alias_lower) in context_lower:
                    return self.bonus_multiplier
        
        return 1.0