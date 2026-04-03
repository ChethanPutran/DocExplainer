# src/core/ml_orchestrator.py
from typing import Dict, Any, List
import logging

from core.document.classifiers.document_classifier import DocumentSectionClassifier
from core.document.models.structure import Document
from core.document.parser.anomaly_detector import DocumentAnomalyDetector
from core.evaluation.evaluators.difficulty_predictor import DifficultyPredictor
from core.explanation_engine.recommenders.ml_recommender import MLRecommendationSystem
from core.knowledge.extraction.canonicalization.concept_clustering import ConceptClusterer
from core.user.services.learning_patterns import LearningPatternAnalyzer

class MLFeatureOrchestrator:
    """Orchestrate all ML features across the system"""
    
    def __init__(self):
        self.anomaly_detector = DocumentAnomalyDetector()
        self.recommender = MLRecommendationSystem()
        self.concept_clusterer = ConceptClusterer()
        self.section_classifier = DocumentSectionClassifier()
        self.difficulty_predictor = DifficultyPredictor()
        self.learning_analyzer = LearningPatternAnalyzer()
        
    def process_document(self, document: Document) -> Dict[str, Any]:
        """Apply all ML features to a document"""
        results = {
            'anomalies': self.anomaly_detector.detect_structural_anomalies(document),
            'section_classifications': {},
            'concept_clusters': None,
            'difficulty_estimates': {}
        }
        
        # Classify each section
        for section in document.sections:
            results['section_classifications'][section.sec_id] = \
                self.section_classifier.classify_section_type(section)
            
            results['difficulty_estimates'][section.sec_id] = \
                self.difficulty_predictor.predict_difficulty(section)
        
        # Cluster concepts if knowledge graph exists
        if hasattr(document, 'knowledge_graph'):
            concepts = document.knowledge_graph.get_all_concepts()
            results['concept_clusters'] = self.concept_clusterer.cluster_concepts(concepts)
        
        return results
    
    def get_recommendations(self, user_id: str, context: Dict) -> List[Dict]:
        """Get personalized recommendations for user"""
        return self.recommender.recommend_content(
            user_id=user_id,
            user_embeddings=self._get_user_embeddings(user_id),
            content_embeddings=self._get_content_embeddings(context),
            top_k=5
        )