from abc import ABC, abstractmethod
import random
import string

from ...base.interfaces import QuestionGenerationStrategy
from ...models.schemas import Question, QuestionOption
from ...models.enums import QuestionType, DifficultyLevel


class BaseQuestionStrategy(QuestionGenerationStrategy, ABC):
    """Base class for question generation strategies"""
    
    def __init__(self, generator):
        self.generator = generator
    
    @abstractmethod
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate a question"""
        pass
    
    @abstractmethod
    def get_question_type(self) -> QuestionType:
        """Get question type"""
        pass
    
    def _generate_id(self) -> str:
        """Generate a question ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_difficulty_adjustments(self, difficulty: DifficultyLevel) -> dict:
        """Get difficulty-based adjustments"""
        adjustments = {
            DifficultyLevel.BEGINNER: {
                'distractor_count': 2,
                'hint_probability': 0.8,
                'explanation_detail': 'simple'
            },
            DifficultyLevel.INTERMEDIATE: {
                'distractor_count': 3,
                'hint_probability': 0.5,
                'explanation_detail': 'moderate'
            },
            DifficultyLevel.ADVANCED: {
                'distractor_count': 4,
                'hint_probability': 0.3,
                'explanation_detail': 'detailed'
            },
            DifficultyLevel.EXPERT: {
                'distractor_count': 5,
                'hint_probability': 0.1,
                'explanation_detail': 'comprehensive'
            }
        }
        return adjustments.get(difficulty, adjustments[DifficultyLevel.INTERMEDIATE])
    
    def _generate_plausible_distractors(self, concept: str, 
                                        correct_answer: str,
                                        num_distractors: int) -> List[str]:
        """Generate plausible wrong answers"""
        # This is a simple implementation - in production, use more sophisticated methods
        distractors = []
        
        # Common misconceptions
        misconceptions = [
            f"the opposite of {correct_answer}",
            f"a simplified version",
            f"an unrelated concept"
        ]
        
        for i in range(min(num_distractors, len(misconceptions))):
            distractors.append(misconceptions[i])
        
        # Fill remaining with generic distractors
        while len(distractors) < num_distractors:
            distractors.append(f"Option {string.ascii_uppercase[len(distractors)]}")
        
        return distractors