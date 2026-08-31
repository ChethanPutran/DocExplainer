# src/core/evaluation/evaluators/difficulty_predictor.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import numpy as np

class DifficultyPredictor:
    """Predict difficulty level of content using regression"""
    
    def __init__(self):
        self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
        self.ridge_regressor = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        
    def extract_difficulty_features(self, content: Any) -> np.ndarray:
        """Extract features for difficulty prediction"""
        features = []
        
        # Text complexity features
        if hasattr(content, 'raw_text'):
            text = content.raw_text
            features.append(len(text))  # Length
            features.append(len(text.split()))  # Word count
            features.append(text.count('.') + text.count('?') + text.count('!'))  # Sentence count
            
            # Average word length
            words = text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            features.append(avg_word_len)
            
            # Technical term density
            tech_terms = ['algorithm', 'function', 'class', 'data', 'model', 'network']
            tech_count = sum(1 for term in tech_terms if term in text.lower())
            features.append(tech_count)
        
        # Structural features
        if hasattr(content, 'paragraphs'):
            features.append(len(content.paragraphs))
            features.append(len(content.subsections) if hasattr(content, 'subsections') else 0)
        
        return np.array(features).reshape(1, -1)
    
    def predict_difficulty(self, content: Any) -> Dict[str, Any]:
        """Predict difficulty level (1-10 scale)"""
        features = self.extract_difficulty_features(content)
        features_scaled = self.scaler.transform(features)
        
        predictions = {
            'random_forest': self.rf_regressor.predict(features_scaled)[0],
            'xgboost': self.xgb_regressor.predict(features_scaled)[0],
            'ridge': self.ridge_regressor.predict(features_scaled)[0]
        }
        
        # Ensemble prediction (weighted average)
        ensemble_score = (
            predictions['random_forest'] * 0.4 +
            predictions['xgboost'] * 0.4 +
            predictions['ridge'] * 0.2
        )
        
        # Clamp to 1-10 range
        ensemble_score = max(1, min(10, ensemble_score))
        
        return {
            'difficulty_score': ensemble_score,
            'difficulty_level': self._score_to_level(ensemble_score),
            'individual_predictions': predictions
        }
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to difficulty level"""
        if score < 3:
            return "Beginner"
        elif score < 5:
            return "Elementary"
        elif score < 7:
            return "Intermediate"
        elif score < 9:
            return "Advanced"
        else:
            return "Expert"