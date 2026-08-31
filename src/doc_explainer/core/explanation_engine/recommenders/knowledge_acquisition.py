# src/core/active_learning/knowledge_acquisition.py
from sklearn.ensemble import RandomForestClassifier
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
import numpy as np

class ActiveKnowledgeLearner:
    """Actively learn which concepts to explain next"""
    
    def __init__(self, n_initial_samples=10):
        self.learner = None
        self.initialized = False
        self.query_strategies = {
            'uncertainty': uncertainty_sampling,
            'margin': margin_sampling,
            'entropy': entropy_sampling
        }
        
    def initialize_learner(self, X_initial: np.ndarray, y_initial: np.ndarray):
        """Initialize active learner with initial labeled data"""
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.learner = ActiveLearner(
            estimator=classifier,
            query_strategy=uncertainty_sampling,
            X_training=X_initial,
            y_training=y_initial
        )
        self.initialized = True
    
    def query_next_concept(self, X_pool: np.ndarray, strategy: str = 'uncertainty') -> int:
        """Query which concept to learn next"""
        if not self.initialized:
            raise ValueError("Learner not initialized")
        
        query_strategy = self.query_strategies.get(strategy, uncertainty_sampling)
        query_idx, _ = query_strategy(self.learner, X_pool)
        return query_idx[0] if len(query_idx) > 0 else None
    
    def teach_concept(self, X: np.ndarray, y: int):
        """Teach the model about a concept"""
        self.learner.teach(X.reshape(1, -1), np.array([y]))
    
    def get_learning_curve(self) -> Dict:
        """Get learning progress metrics"""
        return {
            'n_queries': len(self.learner.X_training),
            'accuracy': self.learner.score(self.learner.X_training, self.learner.y_training),
            'uncertainty_samples': len(self.learner.X_training)
        }