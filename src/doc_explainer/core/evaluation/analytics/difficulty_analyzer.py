from typing import List, Dict, Optional
import numpy as np
from ..models.schemas import Question, EvaluationResult
from ..models.enums import DifficultyLevel


class DifficultyAnalyzer:
    """Analyze question difficulty based on response patterns"""
    
    def __init__(self):
        self.question_stats = {}  # question_id -> stats
    
    def analyze_difficulty(self, question: Question, 
                          results: List[EvaluationResult]) -> Dict:
        """
        Analyze question difficulty based on response results
        
        Returns:
            Dictionary with difficulty metrics
        """
        if not results:
            return {
                'difficulty_rating': 0.5,
                'discrimination_index': 0.0,
                'guess_probability': 0.25,
                'recommended_difficulty': question.difficulty
            }
        
        # Calculate p-value (proportion correct)
        correct_count = sum(1 for r in results if r.is_correct)
        p_value = correct_count / len(results)
        
        # Convert p-value to difficulty rating (0-1, higher = more difficult)
        # p-value of 1.0 (all correct) -> difficulty 0.0
        # p-value of 0.0 (none correct) -> difficulty 1.0
        difficulty_rating = 1.0 - p_value
        
        # Calculate discrimination index (how well question separates high/low performers)
        discrimination = self._calculate_discrimination(results)
        
        # Estimate guess probability based on question type
        guess_prob = self._estimate_guess_probability(question)
        
        # Determine recommended difficulty level
        recommended = self._map_to_difficulty_level(p_value)
        
        # Store stats
        self.question_stats[question.id] = {
            'p_value': p_value,
            'difficulty_rating': difficulty_rating,
            'discrimination': discrimination,
            'attempts': len(results)
        }
        
        return {
            'difficulty_rating': difficulty_rating,
            'p_value': p_value,
            'discrimination_index': discrimination,
            'guess_probability': guess_prob,
            'recommended_difficulty': recommended,
            'total_attempts': len(results)
        }
    
    def _calculate_discrimination(self, results: List[EvaluationResult]) -> float:
        """Calculate item discrimination index"""
        if len(results) < 10:  # Need sufficient samples
            return 0.0
        
        # Sort by total score (if available in metadata)
        # For now, use a simplified approach
        # Split into top 27% and bottom 27%
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        n = len(sorted_results)
        n_group = max(1, int(n * 0.27))
        
        top_group = sorted_results[:n_group]
        bottom_group = sorted_results[-n_group:]
        
        # Proportion correct in each group
        p_top = sum(1 for r in top_group if r.is_correct) / len(top_group)
        p_bottom = sum(1 for r in bottom_group if r.is_correct) / len(bottom_group)
        
        return p_top - p_bottom
    
    def _estimate_guess_probability(self, question: Question) -> float:
        """Estimate probability of guessing correctly"""
        if question.type.value == "true_false":
            return 0.5
        elif question.type.value == "multiple_choice":
            if question.options:
                return 1.0 / len(question.options)
            return 0.25
        else:
            return 0.0  # Very low for open-ended
    
    def _map_to_difficulty_level(self, p_value: float) -> DifficultyLevel:
        """Map p-value to difficulty level"""
        if p_value >= 0.8:
            return DifficultyLevel.BEGINNER
        elif p_value >= 0.6:
            return DifficultyLevel.INTERMEDIATE
        elif p_value >= 0.4:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT
    
    def get_question_stats(self, question_id: str) -> Optional[Dict]:
        """Get stored statistics for a question"""
        return self.question_stats.get(question_id)
    
    def identify_problematic_questions(self, threshold: float = 0.2) -> List[str]:
        """Identify questions with poor discrimination"""
        problematic = []
        for q_id, stats in self.question_stats.items():
            if stats['discrimination'] < threshold:
                problematic.append(q_id)
        return problematic