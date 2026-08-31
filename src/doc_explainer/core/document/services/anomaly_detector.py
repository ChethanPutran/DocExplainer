# src/core/document/parser/anomaly_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import List, Dict, Any
import joblib
from src.core.document.models import Document, Section

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
            'structural_features': features,
            'unusual_section_depth': self._detect_depth_anomalies(document),
            'missing_required_sections': self._detect_missing_sections(document),
            'broken_hierarchy': self._detect_hierarchy_anomalies(document)
        }
        
        return anomalies

    def _extract_structural_features(self, document: Document) -> Dict[str, Any]:
        """Extract basic structural features from a document.

        Returns a dict with counts and depth statistics that can be
        used by anomaly detectors.
        """
        sections = getattr(document, 'sections', []) or []
        num_sections = len(sections)
        titles = [getattr(s, 'title', '') for s in sections]
        paragraphs_per_section = [len(getattr(s, 'paragraphs', []) or []) for s in sections]

        feature_dict = {
            'num_sections': num_sections,
            'avg_paragraphs_per_section': float(np.mean(paragraphs_per_section)) if paragraphs_per_section else 0.0,
            'std_paragraphs_per_section': float(np.std(paragraphs_per_section)) if paragraphs_per_section else 0.0,
            'section_titles': titles,
        }

        return feature_dict

    def _detect_depth_anomalies(self, document: Document) -> List[Dict[str, Any]]:
        """Simple check for unusually deep section nesting based on a `depth` attribute."""
        anomalies = []
        sections = getattr(document, 'sections', []) or []
        for s in sections:
            depth = getattr(s, 'depth', None)
            if depth is not None and depth > 6:  # arbitrary threshold
                anomalies.append({'section': getattr(s, 'title', ''), 'depth': depth})
        return anomalies

    def _detect_missing_sections(self, document: Document) -> List[str]:
        """Check for commonly required sections (e.g., Introduction, Conclusion)."""
        required = {'Introduction', 'Conclusion'}
        sections = getattr(document, 'sections', []) or []
        titles = {getattr(s, 'title', '') for s in sections}
        missing = list(required - titles)
        return missing

    def _detect_hierarchy_anomalies(self, document: Document) -> List[Dict[str, Any]]:
        """Detect basic hierarchy inconsistencies (e.g., sibling with deeper level than parent)."""
        anomalies = []
        sections = getattr(document, 'sections', []) or []
        # If sections have (index, parent_index) attributes, perform a basic check
        for i, s in enumerate(sections):
            parent_idx = getattr(s, 'parent_index', None)
            if parent_idx is not None and (parent_idx >= i or parent_idx < 0):
                anomalies.append({'section': getattr(s, 'title', ''), 'parent_index': parent_idx})
        return anomalies
    
    def detect_content_anomalies(self, sections: List[Section]) -> List[Dict]:
        """Detect unusual content patterns"""
        anomalies = []
        
        for section in sections:
            # Check for unusually short/long paragraphs
            para_lengths = [len(p.text) for p in section.paragraphs]
            if para_lengths:
                mean_len = np.mean(para_lengths)
                std_len = np.std(para_lengths)
                
                for para in section.paragraphs:
                    if abs(len(para.text) - mean_len) > 2 * std_len:
                        anomalies.append({
                            'type': 'paragraph_length_anomaly',
                            'section': section.title,
                            'length': len(para.text),
                            'expected_range': (mean_len - 2*std_len, mean_len + 2*std_len)
                        })
        
        return anomalies