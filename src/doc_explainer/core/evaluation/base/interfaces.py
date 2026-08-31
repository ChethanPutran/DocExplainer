from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.schemas import Question, Quiz, EvaluationResult
from ..models.enums import QuestionType, DifficultyLevel


class QuizGeneratorInterface(ABC):
    """Interface for quiz generation"""
    
    @abstractmethod
    def generate_quiz(self, concepts: List[str], 
                     difficulty: DifficultyLevel = DifficultyLevel.ADAPTIVE,
                     num_questions: int = 5) -> Quiz:
        """Generate a quiz based on concepts"""
        pass
    
    @abstractmethod
    def generate_question(self, concept: str, 
                         question_type: QuestionType,
                         difficulty: DifficultyLevel) -> Question:
        """Generate a single question"""
        pass


class ResponseEvaluatorInterface(ABC):
    """Interface for response evaluation"""
    
    @abstractmethod
    def evaluate_response(self, question: Question, answer: str) -> EvaluationResult:
        """Evaluate user response"""
        pass
    
    @abstractmethod
    def calculate_score(self, correct_count: int, total_count: int) -> float:
        """Calculate score percentage"""
        pass


class LearningGainInterface(ABC):
    """Interface for learning gain calculation"""
    
    @abstractmethod
    def calculate_learning_gain(self, pre_test: Dict, post_test: Dict) -> float:
        """Calculate learning gain from pre/post tests"""
        pass
    
    @abstractmethod
    def calculate_normalized_gain(self, pre_score: float, post_score: float) -> float:
        """Calculate normalized learning gain"""
        pass


class QuestionGenerationStrategy(ABC):
    """Strategy for generating specific question types"""
    
    @abstractmethod
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate a question"""
        pass
    
    @abstractmethod
    def get_question_type(self) -> QuestionType:
        """Get the question type this strategy handles"""
        pass