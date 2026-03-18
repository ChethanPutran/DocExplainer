from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .enums import QuestionType, DifficultyLevel, ResponseCorrectness
import uuid

class QuestionOption(BaseModel):
    """Option for multiple choice questions"""
    text: str
    is_correct: bool = False
    explanation: Optional[str] = None


class Question(BaseModel):
    """Question model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    type: QuestionType
    difficulty: DifficultyLevel
    concept: str
    options: List[QuestionOption] = []
    correct_answer: str
    explanation: Optional[str] = None
    hints: List[str] = []
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    
    def validate_answer(self, answer: str) -> bool:
        """Validate if answer is correct"""
        if self.type == QuestionType.MULTIPLE_CHOICE:
            # For multiple choice, check if answer matches correct option
            for option in self.options:
                if option.is_correct and option.text.lower() == answer.lower():
                    return True
            return False
        elif self.type == QuestionType.TRUE_FALSE:
            return answer.lower() == self.correct_answer.lower()
        else:
            # Simple string matching for other types
            return answer.strip().lower() == self.correct_answer.strip().lower()
    
    def get_correct_option_text(self) -> Optional[str]:
        """Get text of correct option for multiple choice"""
        for option in self.options:
            if option.is_correct:
                return option.text
        return None


class Quiz(BaseModel):
    """Quiz model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: Optional[str] = None
    questions: List[Question]
    difficulty: DifficultyLevel
    concepts: List[str]
    time_limit_minutes: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def total_questions(self) -> int:
        """Get total number of questions"""
        return len(self.questions)
    
    def get_questions_by_concept(self, concept: str) -> List[Question]:
        """Get questions for a specific concept"""
        return [q for q in self.questions if q.concept == concept]
    
    def get_questions_by_difficulty(self, difficulty: DifficultyLevel) -> List[Question]:
        """Get questions by difficulty"""
        return [q for q in self.questions if q.difficulty == difficulty]


class EvaluationResult(BaseModel):
    """Evaluation result model"""
    question_id: str
    user_answer: str
    is_correct: bool
    correctness: ResponseCorrectness
    score: float
    feedback: Optional[str] = None
    time_spent_seconds: Optional[float] = None
    attempts: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuizResult(BaseModel):
    """Quiz result model"""
    quiz_id: str
    user_id: str
    results: List[EvaluationResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_score: float = 0.0
    max_score: float = 0.0
    
    @property
    def percentage(self) -> float:
        """Get percentage score"""
        if self.max_score == 0:
            return 0.0
        return (self.total_score / self.max_score) * 100
    
    @property
    def correct_count(self) -> int:
        """Get number of correct answers"""
        return sum(1 for r in self.results if r.is_correct)
    
    @property
    def time_taken_minutes(self) -> Optional[float]:
        """Get time taken in minutes"""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() / 60
        return None


