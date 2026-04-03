# src/core/document/classifiers/document_classifier.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
import joblib

class DocumentSectionClassifier:
    """Classify document sections using ensemble methods"""
    
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
    def extract_features(self, section: Section) -> np.ndarray:
        """Extract features from section for classification"""
        features = []
        
        # Text-based features
        features.append(len(section.raw_text))
        features.append(len(section.paragraphs))
        features.append(len(section.images))
        features.append(len(section.tables))
        
        # Structural features
        features.append(section.page_start)
        features.append(len(section.subsections))
        
        # Content features
        avg_para_len = np.mean([len(p.raw_text) for p in section.paragraphs]) if section.paragraphs else 0
        features.append(avg_para_len)
        
        # Keyword-based features
        title_words = set(section.title.lower().split())
        features.append(len(title_words))
        
        return np.array(features)
    
    def classify_section_type(self, section: Section) -> Dict[str, float]:
        """Classify section type with confidence scores"""
        features = self.extract_features(section).reshape(1, -1)
        
        predictions = {
            'random_forest': self.rf_classifier.predict_proba(features)[0],
            'svm': self.svm_classifier.predict_proba(features)[0],
            'xgboost': self.xgb_classifier.predict_proba(features)[0],
            'gradient_boosting': self.gb_classifier.predict_proba(features)[0]
        }
        
        # Ensemble voting
        ensemble_pred = self._ensemble_vote(predictions)
        
        return {
            'section_type': self._get_section_type(ensemble_pred),
            'confidence': max(ensemble_pred),
            'individual_predictions': predictions
        }
    
    def detect_technical_content(self, section: Section) -> float:
        """Detect if section contains technical/scientific content"""
        features = self.extract_features(section)
        
        # Use XGBoost for binary classification
        tech_score = self.xgb_classifier.predict_proba(features.reshape(1, -1))[0][1]
        
        return tech_score