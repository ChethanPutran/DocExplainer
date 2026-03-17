import json
import os
from typing import Optional, List
from src.core.user.models.user import User
from src.core.user.models.interaction import UserInteraction
from .serializers import UserSerializer


class UserRepository:
    """Repository for user data persistence"""
    
    def __init__(self, storage_path: str = "data/users/"):
        self.storage_path = storage_path
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_user_path(self, user_id: str) -> str:
        """Get file path for a user"""
        return os.path.join(self.storage_path, f"{user_id}.json")
    
    def save_user(self, user: User) -> User:
        """Save user data"""
        filepath = self._get_user_path(user.user_id)
        
        with open(filepath, 'w') as f:
            json.dump(UserSerializer.serialize_user(user), f, indent=2)
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        filepath = self._get_user_path(user_id)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            return UserSerializer.deserialize_user(data)
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        filepath = self._get_user_path(user_id)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        
        return False
    
    def list_users(self) -> List[str]:
        """List all user IDs"""
        users = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                users.append(filename.replace('.json', ''))
        
        return users
    
    def update_user(self, user: User) -> User:
        """Update user"""
        return self.save_user(user)
    
    def save_interaction(self, user_id: str, interaction: UserInteraction) -> bool:
        """Save user interaction"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.interaction_history.append(interaction)
        self.save_user(user)
        return True
    
    def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Get user interactions"""
        user = self.get_user(user_id)
        if not user:
            return []
        
        return user.interaction_history[-limit:]