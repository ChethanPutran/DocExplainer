from typing import List, Optional
import random

from .base import BaseQuizGenerator
from .strategies.multiple_choice import MultipleChoiceStrategy
from .strategies.true_false import TrueFalseStrategy
from .strategies.fill_blank import FillBlankStrategy
from .strategies.adaptive import AdaptiveStrategy
from ..models.enums import QuestionType, DifficultyLevel
from ..models.schemas import Quiz
from ..base.exceptions import QuizGenerationError


class QuizGenerator(BaseQuizGenerator):
    """Main quiz generator implementation"""
    
    def __init__(self, use_llm: bool = False, llm_wrapper=None):
        self.use_llm = use_llm
        self.llm_wrapper = llm_wrapper
        super().__init__()
    
    def _register_default_strategies(self):
        """Register default question generation strategies"""
        self.register_strategy(MultipleChoiceStrategy(self))
        self.register_strategy(TrueFalseStrategy(self))
        self.register_strategy(FillBlankStrategy(self))
        
        if self.use_llm and self.llm_wrapper:
            self.register_strategy(AdaptiveStrategy(self, self.llm_wrapper))
    
    def generate_quiz_from_knowledge_gaps(self, 
                                         unknown_concepts: List[str],
                                         known_concepts: List[str],
                                         num_questions: int = 5) -> Quiz:
        """
        Generate quiz targeting knowledge gaps
        
        Args:
            unknown_concepts: Concepts the user needs to learn
            known_concepts: Concepts the user already knows
            num_questions: Number of questions to generate
        
        Returns:
            Quiz targeting knowledge gaps
        """
        if not unknown_concepts:
            raise QuizGenerationError("No unknown concepts to assess")
        
        # Mix unknown and known concepts to test connections
        test_concepts = []
        
        # Add unknown concepts (focus areas)
        test_concepts.extend(unknown_concepts[:min(3, len(unknown_concepts))])
        
        # Add some known concepts to test connections
        if known_concepts:
            num_known = min(2, len(known_concepts))
            test_concepts.extend(random.sample(known_concepts, num_known))
        
        # Generate quiz with adaptive difficulty
        return self.generate_quiz(
            concepts=test_concepts,
            difficulty=DifficultyLevel.ADAPTIVE,
            num_questions=num_questions
        )
    
    def generate_mastery_quiz(self, concept: str, 
                             mastery_level: float,
                             num_questions: int = 3) -> Quiz:
        """
        Generate quiz to test mastery of a specific concept
        
        Args:
            concept: The concept to test
            mastery_level: Current mastery level (0.0 to 1.0)
            num_questions: Number of questions
        
        Returns:
            Quiz focused on the concept
        """
        # Adjust difficulty based on mastery
        if mastery_level < 0.3:
            difficulty = DifficultyLevel.BEGINNER
        elif mastery_level < 0.6:
            difficulty = DifficultyLevel.INTERMEDIATE
        elif mastery_level < 0.8:
            difficulty = DifficultyLevel.ADVANCED
        else:
            difficulty = DifficultyLevel.EXPERT
        
        return self.generate_quiz(
            concepts=[concept],
            difficulty=difficulty,
            num_questions=num_questions
        )
    
    def generate_review_quiz(self, concepts: List[str], 
                            num_questions: int = 3) -> Quiz:
        """
        Generate a quick review quiz
        
        Args:
            concepts: Concepts to review
            num_questions: Number of questions
        
        Returns:
            Quick review quiz
        """
        # Review quizzes are simpler
        return Quiz(
            title="Quick Review",
            description=f"Review of {len(concepts)} concepts",
            questions=self._generate_review_questions(concepts, num_questions),
            difficulty=DifficultyLevel.BEGINNER,
            concepts=concepts
        )
    
    def _generate_review_questions(self, concepts: List[str], 
                                  num_questions: int) -> List:
        """Generate simple review questions"""
        questions = []
        
        for i in range(min(num_questions, len(concepts) * 2)):
            concept = random.choice(concepts)
            # Use simpler question types for review
            q_type = random.choice([
                QuestionType.TRUE_FALSE,
                QuestionType.MULTIPLE_CHOICE
            ])
            
            question = self.generate_question(
                concept, q_type, DifficultyLevel.BEGINNER
            )
            questions.append(question)
        
        return questions[:num_questions]