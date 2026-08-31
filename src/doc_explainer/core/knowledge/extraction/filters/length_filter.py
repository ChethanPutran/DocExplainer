from typing import List
from .base import BaseFilterStrategy
from ....knowledge.models.concept import Concept

class LengthFilterStrategy(BaseFilterStrategy):
    """Filter concepts by name length"""
    
    def __init__(self, min_words: int = 1, max_words: int = 5):
        self.min_words = min_words
        self.max_words = max_words
    
    def filter(self, concepts: List[Concept], **kwargs) -> List[Concept]:
        """Keep only concepts with word count in range"""
        filtered = []
        
        for concept in concepts:
            num_words = len(concept.name.split())
            if self.min_words <= num_words <= self.max_words:
                filtered.append(concept)
        
        return filtered