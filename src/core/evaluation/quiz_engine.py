"""
Quiz & Feedback Engine for Doc Explainer

Implements multi-format question generation, adaptive difficulty adjustment,
immediate feedback with explanations, and mastery tracking based on quiz responses.

Key Components:
- QuizEngine: Main orchestrator for quiz sessions
- QuestionGenerator: Manages multi-format question generation
- AdaptiveDifficultyManager: Adjusts difficulty based on mastery
- FeedbackGenerator: Produces immediate, explanatory feedback
- MasteryUpdater: Updates knowledge state based on responses
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import time

from src.core.evaluation.generators.quiz_generator import QuizGenerator
from src.core.evaluation.mastery_estimator import (
    MasteryEstimator,
    InteractionResponse,
)
from src.core.evaluation.models.schemas import (
    Question,
    Quiz,
    QuizSession,
    QuizResponse,
    QuizFeedback,
)
from src.core.evaluation.models.enums import (
    QuestionType,
    DifficultyLevel,
    ResponseCorrectness,
)
from src.core.user.models.knowledge_state import KnowledgeState
from src.core.knowledge.models.concept import Concept

logger = logging.getLogger(__name__)


@dataclass
class DifficultyAdjustmentConfig:
    """Configuration for adaptive difficulty adjustment"""
    easy_threshold: float = 0.3
    medium_threshold: float = 0.7
    use_response_correctness: bool = True
    consecutive_correct_for_increase: int = 2
    consecutive_incorrect_for_decrease: int = 2


class AdaptiveDifficultyManager:
    """
    Manages adaptive difficulty adjustment based on mastery level and response correctness.
    
    Difficulty mapping:
    - Easy (< 0.3 mastery): Basic concept recall, simple definitions
    - Medium (0.3-0.7): Application, synthesis, connections
    - Hard (> 0.7): Analysis, evaluation, complex reasoning
    """
    
    def __init__(self, config: Optional[DifficultyAdjustmentConfig] = None):
        """
        Initialize adaptive difficulty manager.
        
        Args:
            config: Configuration for difficulty adjustment
        """
        self.config = config or DifficultyAdjustmentConfig()
        self.response_streak: Dict[str, Tuple[bool, int]] = {}  # concept -> (is_correct, count)
    
    def get_difficulty_for_mastery(self, mastery_level: float) -> DifficultyLevel:
        """
        Get difficulty level based on mastery.
        
        Args:
            mastery_level: Current mastery probability (0.0-1.0)
            
        Returns:
            Appropriate difficulty level
        """
        if mastery_level < self.config.easy_threshold:
            return DifficultyLevel.BEGINNER
        elif mastery_level < self.config.medium_threshold:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.ADVANCED
    
    def adjust_difficulty(self, concept: str, is_correct: bool) -> Optional[DifficultyLevel]:
        """
        Adjust difficulty based on response correctness.
        
        Tracks consecutive correct/incorrect responses to determine if difficulty
        should be increased or decreased.
        
        Args:
            concept: Concept being assessed
            is_correct: Whether response was correct
            
        Returns:
            New difficulty level if adjustment needed, None otherwise
        """
        if concept not in self.response_streak:
            self.response_streak[concept] = (is_correct, 1)
            return None
        
        prev_correct, count = self.response_streak[concept]
        
        if is_correct == prev_correct:
            count += 1
            self.response_streak[concept] = (is_correct, count)
            
            if is_correct and count >= self.config.consecutive_correct_for_increase:
                self.response_streak[concept] = (is_correct, 0)
                return DifficultyLevel.ADVANCED
            elif not is_correct and count >= self.config.consecutive_incorrect_for_decrease:
                self.response_streak[concept] = (is_correct, 0)
                return DifficultyLevel.BEGINNER
        else:
            self.response_streak[concept] = (is_correct, 1)
        
        return None
    
    def reset_concept_streak(self, concept: str) -> None:
        """Reset difficulty adjustment streak for concept."""
        if concept in self.response_streak:
            del self.response_streak[concept]


class FeedbackGenerator:
    """
    Generates immediate, explanatory feedback with learning resources.
    
    Provides:
    - Correct/incorrect indicator
    - Explanation of why answer is correct
    - Related document sections
    - Hints for incorrect responses
    """
    
    def __init__(self):
        """Initialize feedback generator."""
        self.hint_bank: Dict[str, List[str]] = {}
    
    def generate_feedback(
        self,
        question: Question,
        user_answer: str,
        is_correct: bool,
        mastery_level: float = 0.5,
    ) -> QuizFeedback:
        """
        Generate feedback for a quiz response.
        
        Args:
            question: The question being answered
            user_answer: User's response
            is_correct: Whether response is correct
            mastery_level: Current mastery level for personalization
            
        Returns:
            Feedback object with explanation and guidance
        """
        if is_correct:
            explanation = self._generate_correct_explanation(question, mastery_level)
            hint = None
        else:
            explanation = self._generate_incorrect_explanation(question, user_answer)
            hint = self._get_hint_for_question(question)
        
        related_sections = self._get_related_sections(question)
        next_step = self._generate_next_step(question, is_correct, mastery_level)
        
        return QuizFeedback(
            is_correct=is_correct,
            explanation=explanation,
            hint=hint,
            related_sections=related_sections,
            confidence_score=None,
            next_step=next_step,
        )
    
    def _generate_correct_explanation(
        self, question: Question, mastery_level: float
    ) -> str:
        """Generate explanation for correct answer."""
        base_explanation = (
            question.explanation
            or f"Correct! The answer is: {question.get_correct_option_text() or question.correct_answer}"
        )
        
        if mastery_level > 0.7:
            return base_explanation + "\n\nAdvanced insight: Consider how this concept relates to related topics."
        elif mastery_level < 0.3:
            return base_explanation + "\n\nRemember this concept for future questions."
        else:
            return base_explanation
    
    def _generate_incorrect_explanation(self, question: Question, user_answer: str) -> str:
        """Generate explanation for incorrect answer."""
        correct = question.get_correct_option_text() or question.correct_answer
        return (
            f"Not quite right. The correct answer is: {correct}\n\n"
            f"Explanation: {question.explanation or 'Try reviewing the related material.'}"
        )
    
    def _get_hint_for_question(self, question: Question) -> Optional[str]:
        """Get hint for question."""
        if question.hints:
            return question.hints[0]
        return f"Think about the definition of {question.concept}."
    
    def _get_related_sections(self, question: Question) -> List[str]:
        """Get related document sections for the concept."""
        return [
            f"Concept: {question.concept}",
            f"Difficulty: {question.difficulty.value}",
        ]
    
    def _generate_next_step(
        self, question: Question, is_correct: bool, mastery_level: float
    ) -> Optional[str]:
        """Generate recommendation for next step."""
        if is_correct and mastery_level < 0.7:
            return "Try a harder question to improve your mastery."
        elif not is_correct and mastery_level < 0.3:
            return "Review the concept before attempting more questions."
        return None


class MasteryUpdater:
    """
    Updates knowledge state based on quiz responses using BKT.
    
    Integrates with MasteryEstimator to:
    - Update p_knowledge based on correctness
    - Track confidence changes
    - Update interaction history
    - Trigger curriculum adjustments
    """
    
    def __init__(self, mastery_estimator: MasteryEstimator):
        """
        Initialize mastery updater.
        
        Args:
            mastery_estimator: Estimator for knowledge mastery
        """
        self.mastery_estimator = mastery_estimator
    
    def update_mastery_from_response(
        self,
        knowledge_state: KnowledgeState,
        is_correct: bool,
        response_time_seconds: Optional[float] = None,
    ) -> Tuple[KnowledgeState, float]:
        """
        Update knowledge state based on quiz response.
        
        Args:
            knowledge_state: Current knowledge state
            is_correct: Whether response was correct
            response_time_seconds: Time taken to respond
            
        Returns:
            Tuple of (updated_knowledge_state, confidence_change)
        """
        prev_p_knowledge = knowledge_state.p_knowledge
        
        response = InteractionResponse(
            is_correct=is_correct,
            timestamp=datetime.now(),
            response_time_seconds=response_time_seconds,
            confidence=None,
        )
        
        updated_state = self.mastery_estimator.update_from_response(
            knowledge_state, response
        )
        
        confidence_change = updated_state.p_knowledge - prev_p_knowledge
        
        logger.info(
            f"Mastery updated for {knowledge_state.concept.name}: "
            f"{prev_p_knowledge:.3f} -> {updated_state.p_knowledge:.3f} "
            f"(change: {confidence_change:+.3f})"
        )
        
        return updated_state, confidence_change


class QuizEngine:
    """
    Main orchestrator for quiz sessions with adaptive difficulty,
    immediate feedback, and mastery tracking.
    
    Manages the complete quiz lifecycle:
    1. Question generation with adaptive difficulty
    2. Response evaluation and feedback generation
    3. Mastery tracking and updates
    4. Session persistence
    """
    
    def __init__(
        self,
        quiz_generator: Optional[QuizGenerator] = None,
        mastery_estimator: Optional[MasteryEstimator] = None,
        difficulty_config: Optional[DifficultyAdjustmentConfig] = None,
    ):
        """
        Initialize quiz engine.
        
        Args:
            quiz_generator: Generator for creating quizzes
            mastery_estimator: Estimator for knowledge mastery
            difficulty_config: Configuration for adaptive difficulty
        """
        self.quiz_generator = quiz_generator or QuizGenerator(use_llm=False)
        self.mastery_estimator = mastery_estimator or MasteryEstimator()
        self.difficulty_manager = AdaptiveDifficultyManager(difficulty_config)
        self.feedback_generator = FeedbackGenerator()
        self.mastery_updater = MasteryUpdater(self.mastery_estimator)
        
        self.active_sessions: Dict[str, QuizSession] = {}
    
    def create_adaptive_quiz(
        self,
        user_id: str,
        concepts: List[str],
        knowledge_states: Dict[str, KnowledgeState],
        num_questions: int = 5,
    ) -> Tuple[Quiz, QuizSession]:
        """
        Create an adaptive quiz based on current mastery levels.
        
        Difficulty is set per question based on the user's mastery of the concept.
        
        Args:
            user_id: ID of the user taking the quiz
            concepts: Concepts to be assessed
            knowledge_states: Current knowledge states for concepts
            num_questions: Number of questions to generate
            
        Returns:
            Tuple of (Quiz, QuizSession)
        """
        quiz = self.quiz_generator.generate_quiz(
            concepts=concepts,
            difficulty=DifficultyLevel.ADAPTIVE,
            num_questions=num_questions,
        )
        
        adjusted_quiz = self._adjust_quiz_difficulty(quiz, knowledge_states)
        
        session = QuizSession(
            quiz_id=quiz.id,
            user_id=user_id,
            questions=adjusted_quiz.questions,
        )
        
        self.active_sessions[session.id] = session
        
        logger.info(
            f"Created adaptive quiz session {session.id} for user {user_id} "
            f"with {len(concepts)} concepts"
        )
        
        return adjusted_quiz, session
    
    def _adjust_quiz_difficulty(
        self,
        quiz: Quiz,
        knowledge_states: Dict[str, KnowledgeState],
    ) -> Quiz:
        """
        Adjust quiz questions difficulty based on mastery levels.
        
        Args:
            quiz: Original quiz
            knowledge_states: Knowledge states for concepts
            
        Returns:
            Quiz with adjusted difficulties
        """
        adjusted_questions = []
        
        for question in quiz.questions:
            concept_name = question.concept
            knowledge_state = knowledge_states.get(concept_name)
            
            if knowledge_state:
                mastery = knowledge_state.p_knowledge
                new_difficulty = self.difficulty_manager.get_difficulty_for_mastery(mastery)
                question.difficulty = new_difficulty
            
            adjusted_questions.append(question)
        
        quiz.questions = adjusted_questions
        return quiz
    
    def submit_response(
        self,
        session_id: str,
        question_id: str,
        user_answer: str,
        response_time_seconds: Optional[float] = None,
    ) -> QuizResponse:
        """
        Submit a response to a quiz question.
        
        Evaluates the response, generates feedback, and stores the response.
        
        Args:
            session_id: ID of the quiz session
            question_id: ID of the question
            user_answer: User's answer
            response_time_seconds: Time taken to answer
            
        Returns:
            QuizResponse with evaluation and feedback
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Quiz session {session_id} not found")
        
        question = self._get_question_by_id(session.questions, question_id)
        if not question:
            raise ValueError(f"Question {question_id} not found in session")
        
        is_correct = question.validate_answer(user_answer)
        
        response = QuizResponse(
            question_id=question_id,
            user_answer=user_answer,
            is_correct=is_correct,
            response_time_seconds=response_time_seconds,
        )
        
        feedback = self.feedback_generator.generate_feedback(
            question=question,
            user_answer=user_answer,
            is_correct=is_correct,
        )
        response.feedback = feedback
        
        session.responses.append(response)
        
        logger.info(
            f"Response submitted for question {question_id}: "
            f"{'Correct' if is_correct else 'Incorrect'}"
        )
        
        return response
    
    def complete_session(
        self,
        session_id: str,
        knowledge_states: Dict[str, KnowledgeState],
    ) -> QuizSession:
        """
        Complete a quiz session and update mastery.
        
        Args:
            session_id: ID of the session to complete
            knowledge_states: Knowledge states to update
            
        Returns:
            Updated quiz session with final stats
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Quiz session {session_id} not found")
        
        if session.is_completed:
            logger.warning(f"Session {session_id} is already completed")
            return session
        
        session.completed_at = datetime.now()
        
        session.session_stats = {
            "total_questions": len(session.questions),
            "answered_questions": len(session.responses),
            "correct_count": sum(1 for r in session.responses if r.is_correct),
            "score_percentage": session.score_percentage,
            "time_taken_seconds": (
                session.completed_at - session.started_at
            ).total_seconds(),
        }
        
        mastery_updates = self._update_all_masteries(
            session, knowledge_states
        )
        session.mastery_updates = mastery_updates
        
        logger.info(
            f"Quiz session {session_id} completed. "
            f"Score: {session.score_percentage:.1f}%, "
            f"Mastery updates: {mastery_updates}"
        )
        
        return session
    
    def _update_all_masteries(
        self,
        session: QuizSession,
        knowledge_states: Dict[str, KnowledgeState],
    ) -> Dict[str, float]:
        """
        Update mastery for all concepts in session.
        
        Args:
            session: Quiz session with responses
            knowledge_states: Knowledge states to update
            
        Returns:
            Dictionary of concept -> mastery change
        """
        mastery_updates = {}
        
        response_by_concept: Dict[str, List[QuizResponse]] = {}
        for response in session.responses:
            question = self._get_question_by_id(session.questions, response.question_id)
            if question:
                if question.concept not in response_by_concept:
                    response_by_concept[question.concept] = []
                response_by_concept[question.concept].append(response)
        
        for concept, responses in response_by_concept.items():
            knowledge_state = knowledge_states.get(concept)
            if not knowledge_state:
                continue
            
            prev_mastery = knowledge_state.p_knowledge
            
            for response in responses:
                knowledge_state, _ = self.mastery_updater.update_mastery_from_response(
                    knowledge_state,
                    response.is_correct,
                    response.response_time_seconds,
                )
            
            mastery_change = knowledge_state.p_knowledge - prev_mastery
            mastery_updates[concept] = mastery_change
            
            knowledge_states[concept] = knowledge_state
        
        return mastery_updates
    
    def get_next_question(
        self,
        session_id: str,
        knowledge_states: Dict[str, KnowledgeState],
    ) -> Optional[Question]:
        """
        Get next question in session, with adaptive difficulty.
        
        Args:
            session_id: ID of the session
            knowledge_states: Current knowledge states
            
        Returns:
            Next question or None if session complete
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Quiz session {session_id} not found")
        
        remaining = session.get_remaining_questions()
        if not remaining:
            return None
        
        next_question = remaining[0]
        
        concept_state = knowledge_states.get(next_question.concept)
        if concept_state:
            mastery = concept_state.p_knowledge
            new_difficulty = self.difficulty_manager.get_difficulty_for_mastery(mastery)
            next_question.difficulty = new_difficulty
        
        return next_question
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """
        Get current progress in a quiz session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary with progress information
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Quiz session {session_id} not found")
        
        total = len(session.questions)
        answered = len(session.responses)
        correct = sum(1 for r in session.responses if r.is_correct)
        
        return {
            "session_id": session_id,
            "total_questions": total,
            "answered_questions": answered,
            "remaining_questions": total - answered,
            "correct_count": correct,
            "score_percentage": session.score_percentage,
            "is_completed": session.is_completed,
            "started_at": session.started_at,
            "elapsed_seconds": (datetime.now() - session.started_at).total_seconds(),
        }
    
    def _get_question_by_id(self, questions: List[Question], question_id: str) -> Optional[Question]:
        """Get question by ID from list."""
        return next((q for q in questions if q.id == question_id), None)
    
    def export_session(self, session_id: str) -> Dict[str, Any]:
        """
        Export quiz session as dictionary.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary representation of session
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Quiz session {session_id} not found")
        
        return session.model_dump()
    
    def import_session(self, session_data: Dict[str, Any]) -> QuizSession:
        """
        Import quiz session from dictionary.
        
        Args:
            session_data: Dictionary representation
            
        Returns:
            Imported quiz session
        """
        session = QuizSession(**session_data)
        self.active_sessions[session.id] = session
        return session
