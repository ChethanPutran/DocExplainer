# src/core/ensemble/stacking_ensemble.py
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict

class RobustEnsemble:
    """Ensemble methods for robust predictions"""
    
    def __init__(self):
        # Base learners
        self.base_learners = [
            ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]
        
        # Meta learner
        self.meta_learner = LogisticRegression()
        
        # Stacking ensemble
        self.stacking = StackingClassifier(
            estimators=self.base_learners,
            final_estimator=self.meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        # Voting ensemble
        self.voting = VotingClassifier(
            estimators=self.base_learners,
            voting='soft'
        )
    
    def train_ensemble(self, X_train, y_train):
        """Train both stacking and voting ensembles"""
        self.stacking.fit(X_train, y_train)
        self.voting.fit(X_train, y_train)
    
    def predict_with_confidence(self, X_test) -> Dict:
        """Predict with confidence scores"""
        stacking_proba = self.stacking.predict_proba(X_test)[0]
        voting_proba = self.voting.predict_proba(X_test)
        
        # Weighted average
        final_proba = 0.6 * stacking_proba + 0.4 * voting_proba
        
        predictions = np.argmax(final_proba, axis=1)
        confidence = np.max(final_proba, axis=1)
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': final_proba,
            'stacking_proba': stacking_proba,
            'voting_proba': voting_proba
        }