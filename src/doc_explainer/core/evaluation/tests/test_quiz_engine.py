"""
Tests for Quiz & Feedback Engine

Tests multi-format question generation, adaptive difficulty,
feedback generation, mastery updates, and session persistence.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.core.evaluation.quiz_engine import (
    QuizEngine,
    AdaptiveDifficultyManager,
    DifficultyAdjustmentConfig,
    FeedbackGenerator,
    MasteryUpdater,
)
from src.core.evaluation.models.schemas import (
    Question,
    QuestionOption,
    Quiz,
    QuizSession,
    QuizResponse,
)
from src.core.evaluation.models.enums import (
    QuestionType,
    DifficultyLevel,
)
from src.core.evaluation.mastery_estimator import MasteryEstimator, InteractionResponse
from src.core.evaluation.repository.quiz_repository import QuizRepository
from src.core.user.models.knowledge_state import KnowledgeState
from src.core.knowledge.models.concept import Concept


@pytest.fixture
def sample_concept():
    """Create a sample concept for testing."""
    concept = Concept(name="Python Basics")
    return concept


@pytest.fixture
def sample_question(sample_concept):
    """Create a sample multiple choice question."""
    options = [
        QuestionOption(text="Option 1", is_correct=True),
        QuestionOption(text="Option 2", is_correct=False),
        QuestionOption(text="Option 3", is_correct=False),
    ]
    
    question = Question(
        id="q1",
        text="What is a fundamental concept?",
        type=QuestionType.MULTIPLE_CHOICE,
        difficulty=DifficultyLevel.BEGINNER,
        concept="Python Basics",
        options=options,
        correct_answer="Option 1",
        explanation="Option 1 is correct because...",
        hints=["Think about the definition", "Consider the context"],
    )
    return question


@pytest.fixture
def sample_quiz(sample_question):
    """Create a sample quiz."""
    quiz = Quiz(
        id="quiz1",
        title="Python Basics Quiz",
        description="Test your knowledge of Python basics",
        questions=[sample_question],
        difficulty=DifficultyLevel.BEGINNER,
        concepts=["Python Basics"],
    )
    return quiz


@pytest.fixture
def sample_knowledge_state(sample_concept):
    """Create a sample knowledge state."""
    state = KnowledgeState(
        concept=sample_concept,
        p_knowledge=0.5,
        n_attempts=5,
        n_correct=3,
    )
    return state


@pytest.fixture
def mastery_estimator():
    """Create a mastery estimator."""
    return MasteryEstimator()


@pytest.fixture
def quiz_engine(mastery_estimator):
    """Create a quiz engine."""
    return QuizEngine(mastery_estimator=mastery_estimator)


class TestAdaptiveDifficultyManager:
    """Test adaptive difficulty management."""
    
    def test_difficulty_for_low_mastery(self):
        """Test easy difficulty for low mastery."""
        manager = AdaptiveDifficultyManager()
        difficulty = manager.get_difficulty_for_mastery(0.2)
        assert difficulty == DifficultyLevel.BEGINNER
    
    def test_difficulty_for_medium_mastery(self):
        """Test intermediate difficulty for medium mastery."""
        manager = AdaptiveDifficultyManager()
        difficulty = manager.get_difficulty_for_mastery(0.5)
        assert difficulty == DifficultyLevel.INTERMEDIATE
    
    def test_difficulty_for_high_mastery(self):
        """Test advanced difficulty for high mastery."""
        manager = AdaptiveDifficultyManager()
        difficulty = manager.get_difficulty_for_mastery(0.8)
        assert difficulty == DifficultyLevel.ADVANCED
    
    def test_consecutive_correct_increases_difficulty(self):
        """Test that consecutive correct answers increase difficulty."""
        config = DifficultyAdjustmentConfig(consecutive_correct_for_increase=2)
        manager = AdaptiveDifficultyManager(config)
        
        result1 = manager.adjust_difficulty("concept1", True)
        assert result1 is None
        
        result2 = manager.adjust_difficulty("concept1", True)
        assert result2 == DifficultyLevel.ADVANCED
    
    def test_consecutive_incorrect_decreases_difficulty(self):
        """Test that consecutive incorrect answers decrease difficulty."""
        config = DifficultyAdjustmentConfig(consecutive_incorrect_for_decrease=2)
        manager = AdaptiveDifficultyManager(config)
        
        result1 = manager.adjust_difficulty("concept1", False)
        assert result1 is None
        
        result2 = manager.adjust_difficulty("concept1", False)
        assert result2 == DifficultyLevel.BEGINNER
    
    def test_reset_concept_streak(self):
        """Test resetting difficulty adjustment streak."""
        manager = AdaptiveDifficultyManager()
        manager.adjust_difficulty("concept1", True)
        
        assert "concept1" in manager.response_streak
        manager.reset_concept_streak("concept1")
        assert "concept1" not in manager.response_streak


class TestFeedbackGenerator:
    """Test feedback generation."""
    
    def test_feedback_for_correct_answer(self, sample_question):
        """Test feedback generation for correct answer."""
        generator = FeedbackGenerator()
        feedback = generator.generate_feedback(
            question=sample_question,
            user_answer="Option 1",
            is_correct=True,
            mastery_level=0.5,
        )
        
        assert feedback.is_correct is True
        assert feedback.explanation is not None
        assert "correct" in feedback.explanation.lower()
    
    def test_feedback_for_incorrect_answer(self, sample_question):
        """Test feedback generation for incorrect answer."""
        generator = FeedbackGenerator()
        feedback = generator.generate_feedback(
            question=sample_question,
            user_answer="Option 2",
            is_correct=False,
            mastery_level=0.2,
        )
        
        assert feedback.is_correct is False
        assert feedback.explanation is not None
        assert feedback.hint is not None
    
    def test_feedback_includes_related_sections(self, sample_question):
        """Test that feedback includes related sections."""
        generator = FeedbackGenerator()
        feedback = generator.generate_feedback(
            question=sample_question,
            user_answer="Option 1",
            is_correct=True,
        )
        
        assert len(feedback.related_sections) > 0
        assert sample_question.concept in feedback.related_sections[0]
    
    def test_feedback_advanced_mastery(self, sample_question):
        """Test feedback generation for advanced students."""
        generator = FeedbackGenerator()
        feedback = generator.generate_feedback(
            question=sample_question,
            user_answer="Option 1",
            is_correct=True,
            mastery_level=0.85,
        )
        
        assert feedback.is_correct is True
        assert "Advanced" in feedback.explanation or "advanced" in feedback.explanation


class TestMasteryUpdater:
    """Test mastery updates."""
    
    def test_update_mastery_from_correct_response(
        self, sample_knowledge_state, mastery_estimator
    ):
        """Test mastery update from correct response."""
        updater = MasteryUpdater(mastery_estimator)
        
        initial_mastery = sample_knowledge_state.p_knowledge
        updated_state, change = updater.update_mastery_from_response(
            sample_knowledge_state,
            is_correct=True,
            response_time_seconds=5.0,
        )
        
        assert updated_state.p_knowledge > initial_mastery
        assert change > 0
    
    def test_update_mastery_from_incorrect_response(
        self, sample_knowledge_state, mastery_estimator
    ):
        """Test mastery update from incorrect response."""
        updater = MasteryUpdater(mastery_estimator)
        
        initial_mastery = sample_knowledge_state.p_knowledge
        updated_state, change = updater.update_mastery_from_response(
            sample_knowledge_state,
            is_correct=False,
        )
        
        # BKT should update the mastery state
        # The returned state is the updated state
        assert updated_state is not None
        assert change is not None
    
    def test_update_increases_attempt_count(
        self, sample_knowledge_state, mastery_estimator
    ):
        """Test that update increases attempt count."""
        updater = MasteryUpdater(mastery_estimator)
        
        initial_attempts = sample_knowledge_state.n_attempts
        updated_state, _ = updater.update_mastery_from_response(
            sample_knowledge_state,
            is_correct=True,
        )
        
        assert updated_state.n_attempts == initial_attempts + 1


class TestQuizEngine:
    """Test quiz engine functionality."""
    
    def test_create_adaptive_quiz(self, quiz_engine, sample_concept, sample_knowledge_state):
        """Test creation of adaptive quiz."""
        user_id = "user1"
        concepts = [sample_concept.name]
        knowledge_states = {sample_concept.name: sample_knowledge_state}
        
        with patch.object(quiz_engine.quiz_generator, 'generate_quiz') as mock_gen:
            quiz = Quiz(
                title="Test Quiz",
                questions=[],
                difficulty=DifficultyLevel.ADAPTIVE,
                concepts=concepts,
            )
            mock_gen.return_value = quiz
            
            generated_quiz, session = quiz_engine.create_adaptive_quiz(
                user_id=user_id,
                concepts=concepts,
                knowledge_states=knowledge_states,
                num_questions=5,
            )
            
            assert isinstance(session, QuizSession)
            assert session.user_id == user_id
            assert session.quiz_id == quiz.id
    
    def test_submit_response(self, quiz_engine, sample_question, sample_quiz):
        """Test response submission."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        response = quiz_engine.submit_response(
            session_id=session.id,
            question_id=sample_question.id,
            user_answer="Option 1",
            response_time_seconds=10.0,
        )
        
        assert isinstance(response, QuizResponse)
        assert response.is_correct is True
        assert response.question_id == sample_question.id
        assert response.feedback is not None
    
    def test_submit_incorrect_response(self, quiz_engine, sample_question, sample_quiz):
        """Test incorrect response submission."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        response = quiz_engine.submit_response(
            session_id=session.id,
            question_id=sample_question.id,
            user_answer="Option 2",
            response_time_seconds=5.0,
        )
        
        assert response.is_correct is False
        assert response.feedback is not None
        assert response.feedback.hint is not None
    
    def test_get_next_question(self, quiz_engine, sample_question, sample_quiz, sample_knowledge_state):
        """Test getting next question."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        knowledge_states = {sample_question.concept: sample_knowledge_state}
        next_q = quiz_engine.get_next_question(session.id, knowledge_states)
        
        assert next_q is not None
        assert next_q.id == sample_question.id
    
    def test_get_session_progress(self, quiz_engine, sample_question, sample_quiz):
        """Test getting session progress."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        progress = quiz_engine.get_session_progress(session.id)
        
        assert progress['total_questions'] == 1
        assert progress['answered_questions'] == 0
        assert progress['is_completed'] is False
    
    def test_complete_session(self, quiz_engine, sample_question, sample_quiz, sample_knowledge_state):
        """Test session completion."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        response = quiz_engine.submit_response(
            session_id=session.id,
            question_id=sample_question.id,
            user_answer="Option 1",
        )
        
        knowledge_states = {sample_question.concept: sample_knowledge_state}
        completed_session = quiz_engine.complete_session(session.id, knowledge_states)
        
        assert completed_session.is_completed is True
        assert completed_session.completed_at is not None
        assert 'score_percentage' in completed_session.session_stats
        assert completed_session.score_percentage == 100.0
    
    def test_export_and_import_session(self, quiz_engine, sample_question, sample_quiz):
        """Test session export and import."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        quiz_engine.active_sessions[session.id] = session
        
        exported = quiz_engine.export_session(session.id)
        assert isinstance(exported, dict)
        assert exported['id'] == session.id
        
        imported = quiz_engine.import_session(exported)
        assert imported.id == session.id
        assert len(imported.questions) == 1


class TestQuizRepository:
    """Test quiz repository persistence."""
    
    def test_save_and_load_session(self, sample_quiz):
        """Test saving and loading quiz session."""
        repo = QuizRepository()
        
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        
        assert repo.save_session(session) is True
        loaded = repo.load_session(session.id)
        
        assert loaded is not None
        assert loaded.id == session.id
        assert loaded.user_id == "user1"
    
    def test_delete_session(self, sample_quiz):
        """Test deleting quiz session."""
        repo = QuizRepository()
        
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        
        repo.save_session(session)
        assert repo.delete_session(session.id) is True
        assert repo.load_session(session.id) is None
    
    def test_get_all_user_sessions(self, sample_quiz):
        """Test getting all sessions for a user."""
        repo = QuizRepository()
        
        session1 = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        session2 = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        
        repo.save_session(session1)
        repo.save_session(session2)
        
        sessions = repo.get_all_user_sessions("user1")
        assert len(sessions) >= 2
    
    def test_save_and_get_response(self, sample_question, sample_quiz):
        """Test saving and retrieving responses."""
        repo = QuizRepository()
        
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        
        response = QuizResponse(
            question_id=sample_question.id,
            user_answer="Option 1",
            is_correct=True,
        )
        
        repo.save_response(session.id, response)
        responses = repo.get_responses_for_session(session.id)
        
        assert len(responses) == 1
        assert responses[0].question_id == sample_question.id
    
    def test_track_question_performance(self):
        """Test tracking question performance."""
        repo = QuizRepository()
        
        repo.track_question_performance(
            question_id="q1",
            concept="Python",
            is_correct=True,
            response_time_seconds=5.0,
        )
        repo.track_question_performance(
            question_id="q1",
            concept="Python",
            is_correct=True,
            response_time_seconds=4.0,
        )
        repo.track_question_performance(
            question_id="q1",
            concept="Python",
            is_correct=False,
            response_time_seconds=6.0,
        )
        
        perf = repo.get_question_performance("q1")
        assert perf['attempts'] == 3
        assert perf['correct'] == 2
        assert perf['incorrect'] == 1
        assert perf['accuracy'] == 2/3
    
    def test_get_concept_performance(self):
        """Test getting aggregated concept performance."""
        repo = QuizRepository()
        
        repo.track_question_performance("q1", "Python", True, 5.0)
        repo.track_question_performance("q2", "Python", True, 4.0)
        repo.track_question_performance("q2", "Python", False, 6.0)
        
        perf = repo.get_concept_performance("Python")
        assert perf['total_questions'] == 2
        assert perf['total_attempts'] == 3
        assert perf['total_correct'] == 2
        assert perf['accuracy'] == 2/3
    
    def test_export_all_data(self, sample_quiz):
        """Test exporting all data."""
        repo = QuizRepository()
        
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=sample_quiz.questions,
        )
        repo.save_session(session)
        
        data = repo.export_all_data()
        assert 'sessions' in data
        assert 'responses' in data
        assert 'question_performance' in data


class TestMultiFormatQuestions:
    """Test multi-format question handling."""
    
    def test_multiple_choice_validation(self):
        """Test multiple choice question validation."""
        options = [
            QuestionOption(text="Correct", is_correct=True),
            QuestionOption(text="Wrong", is_correct=False),
        ]
        question = Question(
            text="Which is correct?",
            type=QuestionType.MULTIPLE_CHOICE,
            difficulty=DifficultyLevel.BEGINNER,
            concept="Test",
            options=options,
            correct_answer="Correct",
        )
        
        assert question.validate_answer("Correct") is True
        assert question.validate_answer("Wrong") is False
    
    def test_true_false_validation(self):
        """Test true/false question validation."""
        question = Question(
            text="Is this true?",
            type=QuestionType.TRUE_FALSE,
            difficulty=DifficultyLevel.BEGINNER,
            concept="Test",
            correct_answer="True",
        )
        
        assert question.validate_answer("True") is True
        assert question.validate_answer("False") is False
    
    def test_fill_blank_validation(self):
        """Test fill-in-the-blank validation."""
        question = Question(
            text="The answer is ___",
            type=QuestionType.FILL_BLANK,
            difficulty=DifficultyLevel.BEGINNER,
            concept="Test",
            correct_answer="Python",
        )
        
        assert question.validate_answer("Python") is True
        assert question.validate_answer("python") is True  # Case insensitive
        assert question.validate_answer("JavaScript") is False


class TestSessionStats:
    """Test quiz session statistics."""
    
    def test_score_calculation(self, sample_question, sample_quiz):
        """Test score percentage calculation."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        
        response = QuizResponse(
            question_id=sample_question.id,
            user_answer="Option 1",
            is_correct=True,
        )
        session.responses.append(response)
        
        assert session.score_percentage == 100.0
    
    def test_remaining_questions(self, sample_question, sample_quiz):
        """Test getting remaining questions."""
        session = QuizSession(
            quiz_id=sample_quiz.id,
            user_id="user1",
            questions=[sample_question],
        )
        
        assert len(session.get_remaining_questions()) == 1
        
        response = QuizResponse(
            question_id=sample_question.id,
            user_answer="Option 1",
            is_correct=True,
        )
        session.responses.append(response)
        
        assert len(session.get_remaining_questions()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
