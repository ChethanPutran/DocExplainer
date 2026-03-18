from typing import List, Dict, Optional, Any
import logging

from ..base.exceptions import EvaluationError
from ..generators.quiz_generator import QuizGenerator
from ..evaluators.response_evaluator import ResponseEvaluator
from ..analytics.mastery_tracker import MasteryTracker
from ..analytics.learning_gain import LearningGainCalculator
from ..models.schemas import Question, Quiz, EvaluationResult, QuizResult
from ..models.dataclasses import QuestionAttempt, ConceptMastery
from ..models.enums import DifficultyLevel
from ..config import EvaluationConfig

logger = logging.getLogger(__name__)


class KnowledgeEvaluator:
    """
    Main knowledge evaluator that integrates quiz generation,
    response evaluation, and mastery tracking.
    """
    
    def __init__(self,
                 quiz_generator: Optional[QuizGenerator] = None,
                 response_evaluator: Optional[ResponseEvaluator] = None,
                 mastery_tracker: Optional[MasteryTracker] = None,
                 config: Optional[EvaluationConfig] = None):
        
        self.config = config or EvaluationConfig()
        self.quiz_generator = quiz_generator or QuizGenerator(
            use_llm=self.config.use_llm
        )
        self.response_evaluator = response_evaluator or ResponseEvaluator(
            similarity_threshold=self.config.similarity_threshold,
            enable_partial_credit=self.config.enable_partial_credit
        )
        self.mastery_tracker = mastery_tracker or MasteryTracker(
            decay_rate=self.config.decay_rate
        )
        self.learning_gain_calculator = LearningGainCalculator()
        
        # Store for quizzes and results
        self.quizzes: Dict[str, Quiz] = {}
        self.results: Dict[str, QuizResult] = {}
    
    def generate_quiz(self, concepts: List[str], 
                     difficulty: str = 'adaptive',
                     num_questions: int = 5) -> Quiz:
        """
        Generate a quiz based on concepts
        
        Args:
            concepts: List of concepts to test
            difficulty: Difficulty level ('beginner', 'intermediate', 'advanced', 'adaptive')
            num_questions: Number of questions to generate
        
        Returns:
            Generated Quiz object
        """
        try:
            # Convert string difficulty to enum
            if difficulty == 'adaptive':
                diff_level = DifficultyLevel.ADAPTIVE
            else:
                diff_level = DifficultyLevel(difficulty)
            
            # Generate quiz
            quiz = self.quiz_generator.generate_quiz(
                concepts=concepts,
                difficulty=diff_level,
                num_questions=num_questions
            )
            
            # Store quiz
            self.quizzes[quiz.id] = quiz
            
            logger.info(f"Generated quiz {quiz.id} with {len(quiz.questions)} questions")
            return quiz
            
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            raise EvaluationError(f"Failed to generate quiz: {e}") from e
    
    def evaluate_response(self, question: Question, answer: str) -> EvaluationResult:
        """
        Evaluate a single response
        
        Args:
            question: The question being answered
            answer: User's answer
        
        Returns:
            Evaluation result
        """
        try:
            result = self.response_evaluator.evaluate_response(question, answer)
            
            # Update mastery tracker
            attempt = QuestionAttempt(
                question_id=question.id,
                user_answer=answer,
                is_correct=result.is_correct,
                time_spent_seconds=result.time_spent_seconds or 0,
                attempt_number=result.attempts,
                feedback=result.feedback
            )
            self.mastery_tracker.update_from_attempt(attempt, question)
            
            return result
            
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            raise EvaluationError(f"Failed to evaluate response: {e}") from e
    
    def evaluate_quiz(self, quiz_id: str, answers: Dict[str, str], 
                     user_id: str) -> QuizResult:
        """
        Evaluate all responses in a quiz
        
        Args:
            quiz_id: ID of the quiz
            answers: Dictionary mapping question_id to answer
            user_id: ID of the user taking the quiz
        
        Returns:
            Quiz result with scores and metrics
        """
        if quiz_id not in self.quizzes:
            raise EvaluationError(f"Quiz {quiz_id} not found")
        
        quiz = self.quizzes[quiz_id]
        results = []
        
        for question in quiz.questions:
            if question.id in answers:
                result = self.evaluate_response(question, answers[question.id])
                results.append(result)
        
        # Calculate total score
        total_score = sum(r.score for r in results)
        max_score = len(quiz.questions)
        
        # Create quiz result
        from datetime import datetime
        quiz_result = QuizResult(
            quiz_id=quiz_id,
            user_id=user_id,
            results=results,
            started_at=datetime.now(),  # In practice, track start time
            completed_at=datetime.now(),
            total_score=total_score,
            max_score=float(max_score)
        )
        
        self.results[f"{quiz_id}_{user_id}"] = quiz_result
        
        return quiz_result
    
    def calculate_learning_gain(self, pre_test: Dict, post_test: Dict) -> float:
        """
        Calculate learning gain from pre/post tests
        
        Args:
            pre_test: Pre-test results (dictionary with scores)
            post_test: Post-test results (dictionary with scores)
        
        Returns:
            Normalized learning gain (0-1)
        """
        return self.learning_gain_calculator.calculate_normalized_gain(
            pre_test.get('score', 0),
            post_test.get('score', 0)
        )
    
    def get_concept_mastery(self, concept: str) -> Optional[float]:
        """Get mastery level for a concept"""
        return self.mastery_tracker.get_mastery(concept)
    
    def get_all_mastery(self) -> Dict[str, float]:
        """Get mastery levels for all concepts"""
        return self.mastery_tracker.get_all_mastery()
    
    def get_weakest_concepts(self, limit: int = 3) -> List[str]:
        """Get the weakest concepts"""
        return self.mastery_tracker.get_weakest_concepts(limit)
    
    def generate_remediation_quiz(self, concepts: List[str]) -> Quiz:
        """Generate a quiz targeting weak concepts"""
        # Adjust difficulty based on mastery
        avg_mastery = 0.0
        count = 0
        
        for concept in concepts:
            mastery = self.get_concept_mastery(concept)
            if mastery is not None:
                avg_mastery += mastery
                count += 1
        
        if count > 0:
            avg_mastery /= count
            
            if avg_mastery < 0.3:
                difficulty = DifficultyLevel.BEGINNER
            elif avg_mastery < 0.6:
                difficulty = DifficultyLevel.INTERMEDIATE
            else:
                difficulty = DifficultyLevel.ADVANCED
        else:
            difficulty = DifficultyLevel.BEGINNER
        
        return self.quiz_generator.generate_quiz(
            concepts=concepts,
            difficulty=difficulty,
            num_questions=self.config.remediation_quiz_size
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'total_quizzes': len(self.quizzes),
            'total_evaluations': len(self.results),
            'concepts_tracked': len(self.mastery_tracker.get_all_mastery()),
            'average_mastery': sum(self.mastery_tracker.get_all_mastery().values()) / max(1, len(self.mastery_tracker.get_all_mastery()))
        }