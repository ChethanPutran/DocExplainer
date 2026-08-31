from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import difflib

from ..base.interfaces import ResponseEvaluatorInterface
from ..models.schemas import Question, EvaluationResult
from ..models.enums import ResponseCorrectness
from ..base.exceptions import ResponseEvaluationError

logger = logging.getLogger(__name__)


class BaseResponseEvaluator(ResponseEvaluatorInterface, ABC):
    """Base class for response evaluators"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    @abstractmethod
    def evaluate_response(self, question: Question, answer: str) -> EvaluationResult:
        """Evaluate user response"""
        pass
    
    def calculate_score(self, correct_count: int, total_count: int) -> float:
        """Calculate score percentage"""
        if total_count == 0:
            return 0.0
        return (correct_count / total_count) * 100
    
    def _check_exact_match(self, user_answer: str, correct_answer: str) -> bool:
        """Check for exact match (case-insensitive)"""
        return user_answer.strip().lower() == correct_answer.strip().lower()
    
    def _check_similarity(self, user_answer: str, correct_answer: str) -> float:
        """Calculate similarity ratio between answers"""
        return difflib.SequenceMatcher(None, 
                                       user_answer.lower(), 
                                       correct_answer.lower()).ratio()
    
    def _determine_correctness(self, is_correct: bool, similarity: float) -> ResponseCorrectness:
        """Determine correctness level"""
        if is_correct:
            return ResponseCorrectness.CORRECT
        elif similarity >= self.similarity_threshold:
            return ResponseCorrectness.PARTIALLY_CORRECT
        else:
            return ResponseCorrectness.INCORRECT
    
    def _generate_feedback(self, question: Question, user_answer: str, 
                          is_correct: bool, similarity: float) -> str:
        """Generate feedback for user"""
        if is_correct:
            return f"Correct! {question.explanation}"
        elif similarity >= self.similarity_threshold:
            return f"Almost there! Your answer is close. {question.explanation}"
        else:
            return f"Not quite. {question.explanation}"