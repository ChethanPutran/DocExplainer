from abc import abstractmethod
import hashlib
import re
import numpy as np
import spacy
from transformers import pipeline
import torch 
from typing import List, Literal, Dict
from gliner import GLiNER

class EmbeddingModel:
    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self,model_name: str = DEFAULT_MODEL_NAME) -> None:
        # self.model = SentenceTransformer(model_name)
        self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts).tolist()
        
        text = " ".join(texts)

        # Deterministic lightweight fallback vector for local development.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return np.array([b / 255.0 for b in digest[:16]])

class NERLLM:
    def __init__(self, llm_client = None) -> None:
        self.llm = llm_client

    def extract_concepts(self, text:  list[str]) -> list[str]:
        if not self.llm:
            raise ValueError("LLM client is not provided.")
        
        """Extract concepts using LLM"""
        prompt = f"""
        Extract the key concepts from the following text. 
        Return only the concepts as a comma-separated list.
        
        Text: {text[:2000]}
        
        Concepts:
        """
        
        try:
            response = self.llm.generate(prompt)
            concepts = [c.strip() for c in response.split(',')]
            return concepts
        except:
            return []

class NERRegex():
    def extract_concepts(self, text:  list[str]) -> list[str]:
        """Extract concepts using linguistic patterns"""
        patterns = [
            r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized phrases
            r'(?:"([^"]+)"\s+(?:is|are|means|refers to))',  # Quoted definitions
            r'(?:\b(?:the|a|an)\s+([A-Za-z-]+\s+(?:of|in|for)\s+[A-Za-z-]+))',  # Noun phrases
            r'\b(?:[a-z]{3,}\s){1,2}(?:systems|learning|networks|logic|making|methods)\b'
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.finditer(pattern, " ".join(text))
            for match in matches:
                concept = match.group(1) if match.groups() else match.group(0)
                if concept and len(concept.split()) <= 5:  # Limit to 5 words
                    concepts.append(concept.strip())
        
        return list(set(concepts))


BackendType = Literal["spacy", "keyphrase", "scibert"]


class NERModel:
    def __init__(self, backend: BackendType = "spacy"):
        self.backend = backend
        self.model = None
        self.device = 0 if torch.cuda.is_available() else -1
        self._load_model()

    def _load_model(self):
        print(f"Loading backend: {self.backend}")
        
        
        if self.backend == "spacy":
            try:
                self.model = spacy.load("en_core_web_sm")
            except:
                spacy.cli.download("en_core_web_md")
                self.model = spacy.load("en_core_web_sm")

        elif self.backend == "keyphrase":
            self.model = pipeline(
                task="token-classification",
                model="ml6team/keyphrase-extraction-distilbert-inspec",
                aggregation_strategy="simple",
                device=self.device,
                use_fast=True
            ) 
            
        elif self.backend == "scibert":
            self.model = pipeline(
                task="token-classification",
                model="jsylee/scibert_scivocab_uncased-finetuned-ner",
                aggregation_strategy="simple",
                device=self.device,
                use_fast=True
            )
        elif self.backend == "gliner":
            # Using the 2026-standard small-but-mighty model
            self.model = GLiNER.from_pretrained("numind/NuNER_Zero-span")

        else:
            raise ValueError("Unsupported backend")

        print("Model loaded.\n")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def extract_concepts(self, text: List[str]) -> List[str]:
        full_text = " ".join(text)

        if self.backend == "spacy":
            return self._spacy_extract(full_text)

        elif self.backend in ["keyphrase", "scibert"]:
            return self._transformer_extract(full_text)
        
        elif self.backend == "gliner":
            return self._gliner_extract(full_text)

        else:
            return []

    # --------------------------------------------------
    # Backend Implementations
    # --------------------------------------------------
    def _spacy_extract(self, text: str) -> List[str]:
        doc = self.model(text)
        concepts = []

        for chunk in doc.noun_chunks:
            if chunk.root.pos_ != "PRON":
                clean = " ".join([t.text for t in chunk if not t.is_stop])
                if len(clean) > 2:
                    concepts.append(clean.lower())

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "EVENT"]:
                concepts.append(ent.text.lower())

        return list(set(concepts))

    def _transformer_extract(self, text: str) -> List[str]:
        results = self.model(text)
        return list(set([res["word"].lower() for res in results]))

    def _gliner_extract(self, text: str) -> List[str]:
        # You can define custom labels on the fly!
        labels = ["concept", "algorithm", "mathematical framework", "hardware"]
        entities = self.model.predict_entities(text, labels, threshold=0.5)
        return list(set([ent["text"].lower() for ent in entities]))



class SpacyExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_noun_phrases(self, text: List[str]):
        doc = self.nlp(" ".join(text))
        return list(set([chunk.text.strip() for chunk in doc.noun_chunks]))

    def extract_named_entities(self, text: List[str]):
        doc = self.nlp(" ".join(text))
        return list(set([ent.text.strip() for ent in doc.ents]))
    
class TextModels:
    def __init__(self,llm_client = None) -> None:
        self.llm_client = llm_client
    
    def get_ner_model(self):
        return NERModel()
    
    def get_embedding_model(self):
        return EmbeddingModel()
    
    def get_ner_regex(self):
        return NERRegex()
    
    def get_spacy_model(self):
        return SpacyExtractor()
    
    def get_ner_llm(self):
        if self.llm_client is None:
            return None
        return NERLLM(self.llm_client)
    
    @abstractmethod   
    def extract_concepts(self, text: list[str]) -> list[str]:
        pass


def compare_models(text: List[str]) -> Dict[str, List[str]]:
    outputs = {}
    BACKENDS = ["spacy", "keyphrase"]
    for backend in BACKENDS:
        print(f"Running: {backend}")
        model = NERModel(backend=backend)
        outputs[backend] = model.extract_concepts(text)

    return outputs

if __name__ == "__main__":
    sample_text = [
        "Autonomous systems rely on reinforcement learning to operate under uncertainty.",
        "The perception module transforms raw sensor inputs into structured representations.",
        "We use Stochastic Gradient Descent (SGD) to optimize the loss function in Neural Networks."
    ]

    results = compare_models(sample_text)

    for backend, concepts in results.items():
        print(f"\n=== {backend.upper()} ===")
        for i, c in enumerate(concepts, 1):
            print(f"{i}. {c}")

