from typing import List
from ..base import BaseConceptExtractionStrategy

class NERModelStrategy(BaseConceptExtractionStrategy):
    """Extract concepts using NER model"""
    
    def __init__(self, ner_model):
        self.ner_model = ner_model
    
    def extract(self, text: str) -> List[str]:
        """Extract concepts using NER model"""
        return self.ner_model.extract_concepts(text)