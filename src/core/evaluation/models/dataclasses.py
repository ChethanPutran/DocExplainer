from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class QuestionAttempt:
    """Record of a question attempt"""
    question_id: str
    user_answer: str
    is_correct: bool
    time_spent_seconds: float
    attempt_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptMastery:
    """Mastery level for a concept"""
    concept: str
    mastery_level: float = 0.0  # 0.0 to 1.0
    attempts: int = 0
    correct_attempts: int = 0
    last_attempt: Optional[datetime] = None
    questions_attempted: List[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy"""
        if self.attempts == 0:
            return 0.0
        return self.correct_attempts / self.attempts
    
    def update(self, is_correct: bool):
        """Update mastery based on attempt"""
        self.attempts += 1
        if is_correct:
            self.correct_attempts += 1
        self.last_attempt = datetime.now()
        
        # Update mastery level using simple formula
        # More sophisticated models can be plugged in
        self.mastery_level = self.accuracy * (1 - 0.5 ** self.attempts)


@dataclass
class QuizSession:
    """Session data for a quiz"""
    quiz_id: str
    user_id: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    current_question_index: int = 0
    attempts: List[QuestionAttempt] = field(default_factory=list)
    concept_mastery: Dict[str, ConceptMastery] = field(default_factory=dict)
    
    def add_attempt(self, attempt: QuestionAttempt):
        """Add a question attempt"""
        self.attempts.append(attempt)
        
        # Update concept mastery if we know the concept for this question
        # This would need the question object to get the concept
        # For now, we'll leave this to be handled by the service
    
    def complete(self):
        """Mark quiz as completed"""
        self.completed_at = datetime.now()
    
    @property
    def score(self) -> float:
        """Calculate current score"""
        if not self.attempts:
            return 0.0
        correct = sum(1 for a in self.attempts if a.is_correct)
        return correct / len(self.attempts)
    
    @property
    def time_spent_minutes(self) -> float:
        """Calculate time spent in minutes"""
        end_time = self.completed_at or datetime.now()
        delta = end_time - self.started_at
        return delta.total_seconds() / 60