"""
Quiz Repository for persistent storage of quiz sessions and responses.

Handles:
- Saving/loading quiz sessions
- Tracking question performance
- Querying response history
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
from pathlib import Path

from src.core.evaluation.models.schemas import (
    QuizSession,
    QuizResponse,
    Question,
)

logger = logging.getLogger(__name__)


class QuizRepository:
    """
    Repository for quiz data persistence.
    
    Supports both in-memory and file-based storage for quiz sessions
    and response history.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize quiz repository.
        
        Args:
            storage_path: Optional path for file-based persistence.
                         If None, uses in-memory storage.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.in_memory_sessions: Dict[str, QuizSession] = {}
        self.response_history: Dict[str, List[QuizResponse]] = {}
        self.question_performance: Dict[str, Dict[str, Any]] = {}
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Quiz repository initialized with storage at {storage_path}")
        else:
            logger.info("Quiz repository initialized with in-memory storage")
    
    def save_session(self, session: QuizSession) -> bool:
        """
        Save a quiz session.
        
        Args:
            session: Quiz session to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.in_memory_sessions[session.id] = session
            
            if self.storage_path:
                file_path = self.storage_path / f"session_{session.id}.json"
                session_dict = session.model_dump()
                with open(file_path, 'w') as f:
                    json.dump(session_dict, f, indent=2, default=str)
                logger.info(f"Session {session.id} saved to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[QuizSession]:
        """
        Load a quiz session.
        
        Args:
            session_id: ID of session to load
            
        Returns:
            Quiz session or None if not found
        """
        if session_id in self.in_memory_sessions:
            return self.in_memory_sessions[session_id]
        
        if self.storage_path:
            file_path = self.storage_path / f"session_{session_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    session = QuizSession(**data)
                    self.in_memory_sessions[session_id] = session
                    return session
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
        
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a quiz session.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if session_id in self.in_memory_sessions:
                del self.in_memory_sessions[session_id]
            
            if self.storage_path:
                file_path = self.storage_path / f"session_{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Session {session_id} deleted from {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_all_user_sessions(self, user_id: str) -> List[QuizSession]:
        """
        Get all quiz sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of quiz sessions
        """
        sessions = []
        
        for session in self.in_memory_sessions.values():
            if session.user_id == user_id:
                sessions.append(session)
        
        if self.storage_path:
            for file_path in self.storage_path.glob("session_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if data.get('user_id') == user_id:
                        session = QuizSession(**data)
                        if session.id not in self.in_memory_sessions:
                            sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to load session from {file_path}: {e}")
        
        return sessions
    
    def save_response(self, session_id: str, response: QuizResponse) -> bool:
        """
        Save a quiz response.
        
        Args:
            session_id: Quiz session ID
            response: Quiz response to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"{session_id}:{response.question_id}"
            if session_id not in self.response_history:
                self.response_history[session_id] = []
            
            self.response_history[session_id].append(response)
            
            if self.storage_path:
                file_path = self.storage_path / f"responses_{session_id}.json"
                responses_data = [r.model_dump() for r in self.response_history[session_id]]
                with open(file_path, 'w') as f:
                    json.dump(responses_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save response for session {session_id}: {e}")
            return False
    
    def get_responses_for_session(self, session_id: str) -> List[QuizResponse]:
        """
        Get all responses for a quiz session.
        
        Args:
            session_id: Quiz session ID
            
        Returns:
            List of quiz responses
        """
        if session_id in self.response_history:
            return self.response_history[session_id]
        
        if self.storage_path:
            file_path = self.storage_path / f"responses_{session_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    responses = [QuizResponse(**item) for item in data]
                    self.response_history[session_id] = responses
                    return responses
                except Exception as e:
                    logger.warning(f"Failed to load responses for session {session_id}: {e}")
        
        return []
    
    def track_question_performance(
        self,
        question_id: str,
        concept: str,
        is_correct: bool,
        response_time_seconds: Optional[float] = None,
    ) -> None:
        """
        Track performance metrics for a question.
        
        Args:
            question_id: Question ID
            concept: Concept being tested
            is_correct: Whether response was correct
            response_time_seconds: Time taken to respond
        """
        if question_id not in self.question_performance:
            self.question_performance[question_id] = {
                'concept': concept,
                'attempts': 0,
                'correct': 0,
                'incorrect': 0,
                'total_time_seconds': 0.0,
                'first_attempt_correct': None,
            }
        
        perf = self.question_performance[question_id]
        perf['attempts'] += 1
        
        if is_correct:
            perf['correct'] += 1
            if perf['first_attempt_correct'] is None:
                perf['first_attempt_correct'] = True
        else:
            perf['incorrect'] += 1
            if perf['first_attempt_correct'] is None:
                perf['first_attempt_correct'] = False
        
        if response_time_seconds:
            perf['total_time_seconds'] += response_time_seconds
        
        perf['accuracy'] = perf['correct'] / perf['attempts']
        if perf['attempts'] > 0:
            perf['avg_time_seconds'] = perf['total_time_seconds'] / perf['attempts']
    
    def get_question_performance(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a question.
        
        Args:
            question_id: Question ID
            
        Returns:
            Performance dictionary or None
        """
        return self.question_performance.get(question_id)
    
    def get_concept_performance(self, concept: str) -> Dict[str, Any]:
        """
        Get aggregated performance for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Aggregated performance metrics
        """
        concept_questions = [
            perf for perf in self.question_performance.values()
            if perf.get('concept') == concept
        ]
        
        if not concept_questions:
            return {
                'concept': concept,
                'total_questions': 0,
                'total_attempts': 0,
                'accuracy': 0.0,
            }
        
        total_attempts = sum(q['attempts'] for q in concept_questions)
        total_correct = sum(q['correct'] for q in concept_questions)
        accuracy = total_correct / total_attempts if total_attempts > 0 else 0.0
        
        return {
            'concept': concept,
            'total_questions': len(concept_questions),
            'total_attempts': total_attempts,
            'total_correct': total_correct,
            'accuracy': accuracy,
            'avg_time_seconds': (
                sum(q.get('avg_time_seconds', 0) for q in concept_questions) 
                / len(concept_questions)
            ),
        }
    
    def clear_memory(self) -> None:
        """Clear in-memory cache."""
        self.in_memory_sessions.clear()
        self.response_history.clear()
        logger.info("Quiz repository memory cache cleared")
    
    def export_all_data(self) -> Dict[str, Any]:
        """
        Export all quiz data.
        
        Args:
            
        Returns:
            Dictionary with all quiz data
        """
        return {
            'sessions': {
                sid: s.model_dump() 
                for sid, s in self.in_memory_sessions.items()
            },
            'responses': {
                sid: [r.model_dump() for r in responses]
                for sid, responses in self.response_history.items()
            },
            'question_performance': self.question_performance,
        }
