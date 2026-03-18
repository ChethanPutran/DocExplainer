from abc import ABC, abstractmethod
from typing import List

class BaseConceptExtractionStrategy(ABC):
    """Base class for concept extraction strategies"""
    
    @abstractmethod
    def extract(self, text: str) -> List[str]:
        """Extract concept candidates from text"""
        pass
