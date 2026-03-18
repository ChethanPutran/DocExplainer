import re
import spacy
from typing import List

class TextNormalizer:
    """Normalize concept text using rules and lemmatization"""
    
    def __init__(self, spacy_model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model_name)
    
    def normalize(self, name: str) -> str:
        """Apply normalization rules to concept name"""
        name = name.lower().strip()

        # Remove possessives
        name = re.sub(r"'s\b", "", name)

        # Remove parentheses content
        name = re.sub(r"\(.*?\)", "", name)

        # Remove trailing generic words
        name = re.sub(
            r"\b(model|models|method|methods|approach|approaches)\b", "", name
        )

        # Remove extra spaces
        name = re.sub(r"\s+", " ", name).strip()

        # Lemmatize
        doc = self.nlp(name)
        tokens = []
        for token in doc:
            if not token.is_stop:
                tokens.append(token.lemma_)

        return " ".join(tokens)
    
    def normalize_batch(self, names: List[str]) -> List[str]:
        """Normalize multiple concept names"""
        return [self.normalize(name) for name in names]