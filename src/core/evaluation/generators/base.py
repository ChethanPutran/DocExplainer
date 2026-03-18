from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
import random

from ..base.interfaces import QuizGeneratorInterface, QuestionGenerationStrategy
from ..models.schemas import Question, Quiz
from ..models.enums import QuestionType, DifficultyLevel
from ..base.exceptions import QuizGenerationError

logger = logging.getLogger(__name__)


class BaseQuizGenerator(QuizGeneratorInterface, ABC):
    """Base class for quiz generators"""
    
    def __init__(self):
        self.strategies: Dict[QuestionType, QuestionGenerationStrategy] = {}
        self._register_default_strategies()
    
    @abstractmethod
    def _register_default_strategies(self):
        """Register default question generation strategies"""
        pass
    
    def register_strategy(self, strategy: QuestionGenerationStrategy):
        """Register a question generation strategy"""
        q_type = strategy.get_question_type()
        self.strategies[q_type] = strategy
        logger.info(f"Registered strategy for {q_type.value}")
    
    def get_strategy(self, question_type: QuestionType) -> Optional[QuestionGenerationStrategy]:
        """Get strategy for question type"""
        return self.strategies.get(question_type)
    
    def generate_quiz(self, concepts: List[str], 
                     difficulty: DifficultyLevel = DifficultyLevel.ADAPTIVE,
                     num_questions: int = 5) -> Quiz:
        """Generate a quiz"""
        if not concepts:
            raise QuizGenerationError("No concepts provided for quiz generation")
        
        questions = []
        
        # Determine question types distribution
        if difficulty == DifficultyLevel.ADAPTIVE:
            question_types = self._get_adaptive_distribution(num_questions)
        else:
            question_types = self._get_fixed_distribution(num_questions, difficulty)
        
        # Generate questions
        for i, (concept, q_type) in enumerate(zip(
            self._distribute_concepts(concepts, len(question_types)),
            question_types
        )):
            try:
                question = self.generate_question(concept, q_type, difficulty)
                questions.append(question)
            except Exception as e:
                logger.error(f"Failed to generate question for {concept}: {e}")
                # Continue with other questions
        
        if not questions:
            raise QuizGenerationError("Failed to generate any questions")
        
        # Shuffle questions
        random.shuffle(questions)
        
        return Quiz(
            title=f"Quiz on {', '.join(concepts[:3])}",
            description=f"Assessment covering {len(concepts)} concepts",
            questions=questions,
            difficulty=difficulty,
            concepts=concepts
        )
    
    def generate_question(self, concept: str, 
                         question_type: QuestionType,
                         difficulty: DifficultyLevel) -> Question:
        """Generate a single question using appropriate strategy"""
        strategy = self.get_strategy(question_type)
        if not strategy:
            # Fallback to multiple choice
            strategy = self.get_strategy(QuestionType.MULTIPLE_CHOICE)
        
        if not strategy:
            raise QuizGenerationError(f"No strategy available for {question_type}")
        
        return strategy.generate(concept, difficulty)
    
    def _get_adaptive_distribution(self, num_questions: int) -> List[QuestionType]:
        """Get adaptive question type distribution"""
        # Simple distribution - can be made more sophisticated
        types = [
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.TRUE_FALSE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.FILL_BLANK,
            QuestionType.MULTIPLE_CHOICE
        ]
        
        # Repeat to match num_questions
        result = []
        while len(result) < num_questions:
            result.extend(types)
        
        return result[:num_questions]
    
    def _get_fixed_distribution(self, num_questions: int, 
                                difficulty: DifficultyLevel) -> List[QuestionType]:
        """Get fixed question type distribution based on difficulty"""
        if difficulty == DifficultyLevel.BEGINNER:
            # More simple question types for beginners
            weights = {
                QuestionType.TRUE_FALSE: 0.4,
                QuestionType.MULTIPLE_CHOICE: 0.4,
                QuestionType.FILL_BLANK: 0.2
            }
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            weights = {
                QuestionType.MULTIPLE_CHOICE: 0.5,
                QuestionType.FILL_BLANK: 0.3,
                QuestionType.TRUE_FALSE: 0.2
            }
        else:
            # Advanced: more complex question types
            weights = {
                QuestionType.MULTIPLE_CHOICE: 0.4,
                QuestionType.FILL_BLANK: 0.4,
                QuestionType.SHORT_ANSWER: 0.2
            }
        
        return self._sample_from_weights(weights, num_questions)
    
    def _sample_from_weights(self, weights: Dict[QuestionType, float], 
                            num_samples: int) -> List[QuestionType]:
        """Sample question types based on weights"""
        types = list(weights.keys())
        probs = list(weights.values())
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        
        return random.choices(types, weights=probs, k=num_samples)
    
    def _distribute_concepts(self, concepts: List[str], num_questions: int) -> List[str]:
        """Distribute concepts across questions"""
        if len(concepts) >= num_questions:
            # If we have enough concepts, sample without replacement
            return random.sample(concepts, num_questions)
        else:
            # If we have fewer concepts, repeat them
            result = []
            while len(result) < num_questions:
                result.extend(concepts)
            return result[:num_questions]