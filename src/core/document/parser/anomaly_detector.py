# src/core/document/parser/anomaly_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import List, Dict, Any
import joblib

class DocumentAnomalyDetector:
    """Detect anomalies in document structure and content"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
        
    def detect_structural_anomalies(self, document: Document) -> Dict[str, Any]:
        """Detect anomalies in document structure"""
        features = self._extract_structural_features(document)
        
        # Detect outliers in section hierarchy
        anomalies = {
            'unusual_section_depth': self._detect_depth_anomalies(document),
            'missing_required_sections': self._detect_missing_sections(document),
            'broken_hierarchy': self._detect_hierarchy_anomalies(document)
        }
        
        return anomalies
    
    def detect_content_anomalies(self, sections: List[Section]) -> List[Dict]:
        """Detect unusual content patterns"""
        anomalies = []
        
        for section in sections:
            # Check for unusually short/long paragraphs
            para_lengths = [len(p.raw_text) for p in section.paragraphs]
            if para_lengths:
                mean_len = np.mean(para_lengths)
                std_len = np.std(para_lengths)
                
                for para in section.paragraphs:
                    if abs(len(para.raw_text) - mean_len) > 2 * std_len:
                        anomalies.append({
                            'type': 'paragraph_length_anomaly',
                            'section': section.title,
                            'length': len(para.raw_text),
                            'expected_range': (mean_len - 2*std_len, mean_len + 2*std_len)
                        })
        
        return anomalies