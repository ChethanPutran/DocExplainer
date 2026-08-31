import json
import os
from typing import Any, Optional, Dict, List
from datetime import datetime

from ..models.user import User
from ..models.interaction import UserInteraction
from abc import ABC, abstractmethod


class BaseUserRepository(ABC):
    """Repository for user data persistence"""

    @abstractmethod
    def save_user(self, user: User) -> User:
        """Save user data"""
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        pass

    @abstractmethod
    def list_users(self) -> List[str]:
        """List all user IDs"""
        pass

    @abstractmethod
    def update_user(self, user: User) -> User:
        """Update existing user"""
        pass

    @abstractmethod
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        pass

    @abstractmethod
    def save_interaction(self, user_id: str, interaction: UserInteraction) -> bool:
        """Save user interaction"""
        pass

    @abstractmethod
    def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Get user interactions"""
        pass

    @abstractmethod
    def get_interactions_by_date(self, user_id: str, start_date: datetime, end_date: datetime) -> List[UserInteraction]:
        """Get interactions within date range"""
        pass

    @abstractmethod
    def delete_interactions(self, user_id: str, before_date: datetime) -> int:
        """Delete interactions older than date"""
        pass

    # Search methods
    @abstractmethod
    def find_users_by_concept(self, concept_name: str, min_knowledge: float = 0.7) -> List[str]:
        """Find users who know a specific concept above threshold"""
        pass

    @abstractmethod
    def find_users_by_interaction_count(self, min_interactions: int = 10) -> List[str]:
        """Find users with at least min_interactions"""
        pass

    @abstractmethod
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user"""
        pass

    @abstractmethod
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all users"""
        pass

    @abstractmethod
    def export_user_data(self, user_id: str, export_path: str) -> bool:
        """Export all user data to a single file"""
        pass

    @abstractmethod
    def import_user_data(self, import_path: str) -> Optional[User]:
        """Import user data from file"""
        pass

    @abstractmethod
    def clear_cache(self):
        """Clear in-memory cache"""
        pass

    @abstractmethod
    def reload_user(self, user_id: str) -> Optional[User]:
        """Force reload user from disk"""
        pass
