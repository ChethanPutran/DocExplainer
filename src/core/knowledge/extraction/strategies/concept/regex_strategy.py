from typing import List
import re
from ..base import BaseConceptExtractionStrategy

class RegexNERStrategy(BaseConceptExtractionStrategy):
    """Extract concepts using regex patterns"""
    
    def __init__(self, regex_patterns):
        self.regex_patterns = regex_patterns
    
    def extract(self, text: str) -> List[str]:
        """Extract concepts using regex patterns"""
        return self.regex_patterns.extract_concepts(text)