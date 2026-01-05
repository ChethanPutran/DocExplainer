import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'
    def __init__(self,model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
    def encode(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()