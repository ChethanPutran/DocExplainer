from typing import Optional, Dict, Any
import logging

from src.core.user.user_manager import UserManager
from src.core.user.models.user import User
from src.core.user.models.knowledge_state import UserKnowledgeState


class UserService:
    """Service for user operations"""
    
    def __init__(self, user_manager_factory, logger=None):
        self.user_manager_factory = user_manager_factory
        self.user_managers: Dict[str, UserManager] = {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def get_user_manager(self, user_id: str) -> UserManager:
        """Get user manager for user ID"""
        if user_id not in self.user_managers:
            self.user_managers[user_id] = self.user_manager_factory(user_id)
        return self.user_managers[user_id]
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user_manager = self.get_user_manager(user_id)
        return user_manager.get_user()
    
    def get_user_knowledge(self, user_id: str) -> UserKnowledgeState:
        """Get user knowledge state"""
        user_manager = self.get_user_manager(user_id)
        return user_manager.get_user_knowledge()
    
    def update_user_knowledge(self, user_id: str, knowledge_update: Dict[str, Any]):
        """Update user knowledge"""
        user_manager = self.get_user_manager(user_id)
        user_manager.update_user_knowledge(knowledge_update)
    
    def get_user_confidence(self, user_id: str, concept_name: str) -> float:
        """Get user confidence for a concept"""
        user_manager = self.get_user_manager(user_id)
        return user_manager.user_confidence(concept_name)