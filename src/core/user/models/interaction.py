from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class UserInteraction:
    """Stores user interaction data"""
    subject: str = ""
    level: str = ""
    mastery: float = 0.0
    last_seen: Optional[datetime] = None
    time_spent: float = 0.0
    quiz_response: str = ""
    questions_asked: List[str] = field(default_factory=list)
    explanation_depth_requested: str = ""
    source: str = ""
    correct: Optional[bool] = None
    
    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'subject': self.subject,
            'level': self.level,
            'mastery': self.mastery,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'time_spent': self.time_spent,
            'quiz_response': self.quiz_response,
            'questions_asked': self.questions_asked,
            'explanation_depth_requested': self.explanation_depth_requested,
            'source': self.source,
            'correct': self.correct
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserInteraction':
        """Create from dictionary"""
        interaction = cls(
            subject=data.get('subject', ''),
            level=data.get('level', ''),
            mastery=data.get('mastery', 0.0),
            time_spent=data.get('time_spent', 0.0),
            quiz_response=data.get('quiz_response', ''),
            questions_asked=data.get('questions_asked', []),
            explanation_depth_requested=data.get('explanation_depth_requested', ''),
            source=data.get('source', ''),
            correct=data.get('correct')
        )
        
        last_seen = data.get('last_seen')
        if last_seen:
            interaction.last_seen = datetime.fromisoformat(last_seen)
        
        return interaction