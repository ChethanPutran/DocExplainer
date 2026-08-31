from typing import List
from .base import BaseFilterStrategy
from ....knowledge.models.concept import Concept

class SubsetPrunerStrategy(BaseFilterStrategy):
    """Remove shorter concepts that are substrings of longer, higher-ranked ones"""
    
    def filter(self, concepts: List[Concept], top: int = 5) -> List[Concept]:
        """Prune subset concepts"""
        if not concepts:
            return []
        
        # Sort by length descending
        concepts_sorted = sorted(concepts, key=lambda x: len(x.name), reverse=True)
        
        final = []
        for i, concept in enumerate(concepts_sorted):
            is_subset = False
            for j, other in enumerate(concepts_sorted):
                if i != j and concept.name in other.name and concept.score < other.score:
                    is_subset = True
                    break
            if not is_subset:
                final.append(concept)
        
        # Sort by score and return top
        final.sort(key=lambda x: x.score, reverse=True)
        return final[:top]