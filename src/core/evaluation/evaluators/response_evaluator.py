from typing import Optional
import time

from .base import BaseResponseEvaluator
from ..models.schemas import Question, EvaluationResult
from ..models.enums import ResponseCorrectness
from ..base.exceptions import ResponseEvaluationError


class ResponseEvaluator(BaseResponseEvaluator):
    """Main response evaluator implementation"""
    
    def __init__(self, similarity_threshold: float = 0.8, 
                 enable_partial_credit: bool = True):
        super().__init__(similarity_threshold)
        self.enable_partial_credit = enable_partial_credit
    
    def evaluate_response(self, question: Question, answer: str, 
                         time_spent_seconds: Optional[float] = None,
                         attempt_number: int = 1) -> EvaluationResult:
        """Evaluate user response"""
        if not answer:
            raise ResponseEvaluationError("Empty answer provided")
        
        start_time = time.time()
        
        # Check for exact match
        exact_match = self._check_exact_match(answer, question.correct_answer)
        
        # Calculate similarity
        similarity = self._check_similarity(answer, question.correct_answer)
        
        # Determine correctness
        if exact_match:
            is_correct = True
            correctness = ResponseCorrectness.CORRECT
            score = 1.0
        elif similarity >= self.similarity_threshold and self.enable_partial_credit:
            is_correct = False
            correctness = ResponseCorrectness.PARTIALLY_CORRECT
            score = similarity  # Partial credit based on similarity
        else:
            is_correct = False
            correctness = ResponseCorrectness.INCORRECT
            score = 0.0
        
        # Generate feedback
        feedback = self._generate_feedback(question, answer, is_correct, similarity)
        
        # Calculate time spent if not provided
        if time_spent_seconds is None:
            time_spent_seconds = time.time() - start_time
        
        return EvaluationResult(
            question_id=question.id,
            user_answer=answer,
            is_correct=is_correct,
            correctness=correctness,
            score=score,
            feedback=feedback,
            time_spent_seconds=time_spent_seconds,
            attempts=attempt_number,
            metadata={
                "similarity_score": similarity,
                "exact_match": exact_match,
                "partial_credit_enabled": self.enable_partial_credit
            }
        )
    
    def evaluate_batch(self, question: Question, answers: list) -> list:
        """Evaluate multiple answers to the same question"""
        results = []
        for i, answer in enumerate(answers):
            result = self.evaluate_response(
                question, 
                answer, 
                attempt_number=i+1
            )
            results.append(result)
        return results
    
    def evaluate_quiz(self, questions: list, answers: dict) -> dict:
        """Evaluate all answers in a quiz"""
        results = {}
        total_score = 0.0
        max_score = len(questions)
        
        for question in questions:
            if question.id in answers:
                result = self.evaluate_response(
                    question, 
                    answers[question.id]
                )
                results[question.id] = result
                total_score += result.score
        
        return {
            "results": results,
            "total_score": total_score,
            "percentage": self.calculate_score(total_score, max_score),
            "correct_count": sum(1 for r in results.values() if r.is_correct),
            "total_questions": len(questions)
        }