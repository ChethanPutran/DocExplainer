# src/core/nlp/advanced_processor.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
import numpy as np

class AdvancedNLPProcessor:
    """Advanced NLP features for document understanding"""
    
    def __init__(self):
        # Zero-shot classification for dynamic topic detection
        self.zero_shot_classifier = pipeline("zero-shot-classification", 
                                             model="facebook/bart-large-mnli")
        
        # Summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Question answering
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        # Sentiment analysis for user feedback
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Named entity recognition for concept extraction
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
        
    def extract_topics_lda(self, documents: List[str], n_topics: int = 10) -> Dict:
        """Extract topics using Latent Dirichlet Allocation"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"topic_{topic_idx}"] = {
                'words': top_words,
                'weight': float(topic.sum())
            }
        
        return topics
    
    def zero_shot_topic_classification(self, text: str, candidate_labels: List[str]) -> Dict:
        """Classify text into dynamic topics without training"""
        result = self.zero_shot_classifier(text, candidate_labels)
        return {
            'labels': result['labels'],
            'scores': result['scores'],
            'dominant_topic': result['labels'][0]
        }
    
    def extract_relationships_bert(self, text: str) -> List[Dict]:
        """Extract entity relationships using BERT"""
        entities = self.ner_pipeline(text)
        
        # Group entities by their positions
        relationships = []
        for i, entity in enumerate(entities[:-1]):
            next_entity = entities[i + 1]
            
            # Check if entities are close (potential relationship)
            if next_entity['start'] - entity['end'] < 50:
                relationships.append({
                    'entity1': entity['word'],
                    'entity1_type': entity['entity'],
                    'entity2': next_entity['word'],
                    'entity2_type': next_entity['entity'],
                    'distance': next_entity['start'] - entity['end']
                })
        
        return relationships
    
    def generate_embeddings_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings using sentence transformers"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings