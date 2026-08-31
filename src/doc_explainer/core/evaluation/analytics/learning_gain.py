from typing import Dict, Optional, List
import numpy as np

from ..base.interfaces import LearningGainInterface
from ..base.exceptions import LearningGainError

class LearningGainCalculator(LearningGainInterface):
    """Calculate learning gains from pre/post tests"""
    
    def calculate_learning_gain(self, pre_test: Dict, post_test: Dict) -> float:
        """
        Calculate absolute learning gain
        
        Args:
            pre_test: Dictionary with 'score' key (0-100)
            post_test: Dictionary with 'score' key (0-100)
        
        Returns:
            Absolute gain (post_score - pre_score)
        """
        pre_score = self._extract_score(pre_test)
        post_score = self._extract_score(post_test)
        
        return post_score - pre_score
    
    def calculate_normalized_gain(self, pre_score: float, post_score: float) -> float:
        """
        Calculate normalized learning gain (Hake's formula)
        
        Normalized gain = (post% - pre%) / (100% - pre%)
        
        Args:
            pre_score: Pre-test score (0-100)
            post_score: Post-test score (0-100)
        
        Returns:
            Normalized gain (0-1)
        """
        if pre_score < 0 or pre_score > 100 or post_score < 0 or post_score > 100:
            raise LearningGainError("Scores must be between 0 and 100")
        
        if pre_score == 100:
            return 0.0  # Already at maximum
        
        gain = (post_score - pre_score) / (100 - pre_score)
        return max(0.0, min(1.0, gain))  # Clamp to [0,1]
    
    def calculate_effect_size(self, pre_test: Dict, post_test: Dict) -> float:
        """
        Calculate effect size (Cohen's d)
        
        d = (mean_post - mean_pre) / pooled_std
        """
        pre_scores = self._extract_scores_list(pre_test)
        post_scores = self._extract_scores_list(post_test)
        
        if len(pre_scores) < 2 or len(post_scores) < 2:
            return 0.0
        
        mean_pre = np.mean(pre_scores)
        mean_post = np.mean(post_scores)
        std_pre = np.std(pre_scores, ddof=1)
        std_post = np.std(post_scores, ddof=1)
        
        # Pooled standard deviation
        n_pre = len(pre_scores)
        n_post = len(post_scores)
        pooled_std = np.sqrt(((n_pre - 1) * std_pre**2 + (n_post - 1) * std_post**2) / 
                            (n_pre + n_post - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean_post - mean_pre) / pooled_std
    
    def calculate_relative_gain(self, pre_test: Dict, post_test: Dict) -> float:
        """
        Calculate relative gain (percentage improvement)
        
        relative_gain = (post - pre) / pre * 100
        """
        pre_score = self._extract_score(pre_test)
        post_score = self._extract_score(post_test)
        
        if pre_score == 0:
            return float('inf') if post_score > 0 else 0.0
        
        return ((post_score - pre_score) / pre_score) * 100
    
    def _extract_score(self, test_data: Dict) -> float:
        """Extract score from test data"""
        if 'score' in test_data:
            return float(test_data['score'])
        elif 'percentage' in test_data:
            return float(test_data['percentage'])
        elif 'results' in test_data and isinstance(test_data['results'], list):
            # Calculate from results
            correct = sum(1 for r in test_data['results'] if r.get('is_correct', False))
            total = len(test_data['results'])
            if total > 0:
                return (correct / total) * 100
        
        raise LearningGainError("Could not extract score from test data")
    
    def _extract_scores_list(self, test_data: Dict) -> List[float]:
        """Extract list of scores from test data"""
        if 'scores' in test_data and isinstance(test_data['scores'], list):
            return [float(s) for s in test_data['scores']]
        elif 'results' in test_data and isinstance(test_data['results'], list):
            return [1.0 if r.get('is_correct', False) else 0.0 
                   for r in test_data['results']]
        else:
            # Return single score as list
            return [self._extract_score(test_data)]


