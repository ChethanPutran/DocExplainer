from typing import List, Dict, Any
import numpy as np
from ..models.schemas import EvaluationResult, QuizResult
from ..models.enums import EvaluationMetric


class EvaluationMetrics:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def accuracy(results: List[EvaluationResult]) -> float:
        """Calculate accuracy"""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.is_correct)
        return correct / len(results)
    
    @staticmethod
    def average_score(results: List[EvaluationResult]) -> float:
        """Calculate average score"""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)
    
    @staticmethod
    def average_time_per_question(results: List[EvaluationResult]) -> float:
        """Calculate average time per question"""
        times = [r.time_spent_seconds for r in results if r.time_spent_seconds]
        if not times:
            return 0.0
        return sum(times) / len(times)
    
    @staticmethod
    def difficulty_distribution(results: List[EvaluationResult], 
                               questions: dict) -> Dict[str, float]:
        """Calculate accuracy by difficulty"""
        difficulty_scores = {}
        
        for result in results:
            question = questions.get(result.question_id)
            if question and question.difficulty:
                diff = question.difficulty.value
                if diff not in difficulty_scores:
                    difficulty_scores[diff] = {"correct": 0, "total": 0}
                
                difficulty_scores[diff]["total"] += 1
                if result.is_correct:
                    difficulty_scores[diff]["correct"] += 1
        
        # Calculate percentages
        for diff in difficulty_scores:
            total = difficulty_scores[diff]["total"]
            if total > 0:
                difficulty_scores[diff] = difficulty_scores[diff]["correct"] / total
            else:
                difficulty_scores[diff] = 0.0
        
        return difficulty_scores
    
    @staticmethod
    def concept_mastery(results: List[EvaluationResult], 
                       questions: dict) -> Dict[str, float]:
        """Calculate mastery by concept"""
        concept_scores = {}
        
        for result in results:
            question = questions.get(result.question_id)
            if question and question.concept:
                concept = question.concept
                if concept not in concept_scores:
                    concept_scores[concept] = {"score": 0.0, "count": 0}
                
                concept_scores[concept]["score"] += result.score
                concept_scores[concept]["count"] += 1
        
        # Calculate averages
        for concept in concept_scores:
            count = concept_scores[concept]["count"]
            if count > 0:
                concept_scores[concept] = concept_scores[concept]["score"] / count
            else:
                concept_scores[concept] = 0.0
        
        return concept_scores
    
    @staticmethod
    def get_all_metrics(results: List[EvaluationResult], 
                       questions: dict) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            EvaluationMetric.ACCURACY.value: EvaluationMetrics.accuracy(results),
            "average_score": EvaluationMetrics.average_score(results),
            "average_time": EvaluationMetrics.average_time_per_question(results),
            "difficulty_distribution": EvaluationMetrics.difficulty_distribution(results, questions),
            "concept_mastery": EvaluationMetrics.concept_mastery(results, questions),
            "total_questions": len(results),
            "correct_count": sum(1 for r in results if r.is_correct)
        }