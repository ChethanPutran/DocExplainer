# src/core/nlp/advanced_processor.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from transformers import pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


class AdvancedProcessor(ABC):
    """Abstract interface for advanced NLP document processing."""

    @abstractmethod
    def extract_topics_lda(self, documents: List[str], n_topics: int = 10) -> Dict:
        """Extract topics using Latent Dirichlet Allocation."""

    @abstractmethod
    def zero_shot_topic_classification(self, text: str, candidate_labels: List[str]) -> Dict:
        """Classify text into dynamic topics without training."""

    @abstractmethod
    def extract_relationships_bert(self, text: str) -> List[Dict]:
        """Extract entity relationships using BERT."""

    @abstractmethod
    def generate_embeddings_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings using sentence transformers."""


class AdvancedNLPProcessor(AdvancedProcessor):
    """Advanced NLP features for document understanding"""
    
    def __init__(self):
        # Zero-shot classification for dynamic topic detection
        # use Any typed attributes to avoid strict type-checker issues with transformers stubs
        self.zero_shot_classifier: Any = hf_pipeline("zero-shot-classification", 
                             model="facebook/bart-large-mnli")
        
        # Summarization pipeline
        self.summarizer: Any = hf_pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Question answering
        self.qa_pipeline: Any = hf_pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        # Sentiment analysis for user feedback
        # avoid annotating task variable to satisfy some type checkers
        sentiment_task = "sentiment-analysis"
        self.sentiment_analyzer: Any = hf_pipeline(task=sentiment_task)
        
        # Named entity recognition for concept extraction
        self.ner_pipeline: Any = hf_pipeline("ner", model="dslim/bert-base-NER")
        
    def extract_topics_lda(self, documents: List[str], n_topics: int = 10) -> Dict:
        """Extract topics using Latent Dirichlet Allocation."""
        clean_documents = [doc.strip() for doc in documents if isinstance(doc, str) and doc.strip()]
        if not clean_documents:
            return {}

        if len(clean_documents) == 1:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            matrix = vectorizer.fit_transform(clean_documents)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(matrix.sum(axis=0)).ravel()
            top_indices = np.argsort(scores)[-min(10, len(feature_names)):][::-1]
            return {
                'topic_0': {
                    'words': [feature_names[i] for i in top_indices],
                    'weight': float(scores[top_indices].sum())
                }
            }

        n_components = max(1, min(int(n_topics), len(clean_documents)))
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(clean_documents)

        try:
            lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
            lda.fit(doc_term_matrix)
        except ValueError:
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(doc_term_matrix.mean(axis=0)).ravel()
            top_indices = np.argsort(scores)[-min(10, len(feature_names)):][::-1]
            return {
                'topic_0': {
                    'words': [feature_names[i] for i in top_indices],
                    'weight': float(scores[top_indices].sum())
                }
            }

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10:-1]
            top_words = [feature_names[i] for i in top_words_idx if i < len(feature_names)]
            topics[f"topic_{topic_idx}"] = {
                'words': top_words,
                'weight': float(topic.sum())
            }

        return topics
    
    def zero_shot_topic_classification(self, text: str, candidate_labels: List[str]) -> Dict:
        """Classify text into dynamic topics without training."""
        if not text or not candidate_labels:
            return {
                'labels': [],
                'scores': [],
                'dominant_topic': None
            }

        result: Any = self.zero_shot_classifier(text, candidate_labels)
        labels = []
        scores = []

        if isinstance(result, dict):
            labels = result.get('labels') or []
            scores = result.get('scores') or []
        elif isinstance(result, (list, tuple)):
            parsed = []
            for item in result:
                if isinstance(item, dict):
                    parsed.append(item)
            if parsed:
                labels = [item.get('label') for item in parsed if isinstance(item, dict) and item.get('label') is not None]
                scores = [item.get('score') for item in parsed if isinstance(item, dict) and item.get('score') is not None]

        return {
            'labels': labels,
            'scores': scores,
            'dominant_topic': labels[0] if labels else None
        }
    
    def extract_relationships_bert(self, text: str) -> List[Dict]:
        """Extract entity relationships using BERT."""
        if not text:
            return []

        entities = self.ner_pipeline(text)
        if isinstance(entities, dict):
            entities = entities.get('entities', []) or entities.get('result', []) or []
        if not entities:
            return []

        normalized_entities = []
        for entity in entities:
            if isinstance(entity, dict):
                word = entity.get('word') or entity.get('entity_group') or ''
                label = entity.get('entity') or entity.get('entity_group') or 'UNKNOWN'
                start = entity.get('start', 0)
                end = entity.get('end', start)
                normalized_entities.append({
                    'word': word,
                    'entity': label,
                    'start': start,
                    'end': end
                })

        normalized_entities.sort(key=lambda item: item.get('start', 0))
        relationships = []
        for i in range(len(normalized_entities) - 1):
            entity = normalized_entities[i]
            next_entity = normalized_entities[i + 1]
            distance = next_entity.get('start', 0) - entity.get('end', 0)
            if 0 <= distance < 50:
                relationships.append({
                    'entity1': entity.get('word', ''),
                    'entity1_type': entity.get('entity', 'UNKNOWN'),
                    'entity2': next_entity.get('word', ''),
                    'entity2_type': next_entity.get('entity', 'UNKNOWN'),
                    'distance': distance
                })

        return relationships
    
    def generate_embeddings_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings using sentence transformers."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        normalized_texts = [text.strip() if isinstance(text, str) else "" for text in texts]
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(normalized_texts, convert_to_numpy=True)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(normalized_texts)
            return matrix.toarray().astype(np.float32)