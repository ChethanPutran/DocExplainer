from typing import List
from ..base import BaseConceptExtractionStrategy

class SpacyNounPhraseStrategy(BaseConceptExtractionStrategy):
    """Extract noun phrases using spaCy"""
    
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model
    
    def extract(self, text: str) -> List[str]:
        """Extract noun phrases from text"""
        return self.spacy_model.extract_noun_phrases(text)

class SpacyNERStrategy(BaseConceptExtractionStrategy):
    """Extract named entities using spaCy"""
    
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model
    
    def extract(self, text: str) -> List[str]:
        """Extract named entities from text"""
        return self.spacy_model.extract_named_entities(text)