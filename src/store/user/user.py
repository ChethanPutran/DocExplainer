from datetime import datetime
import json
import os
from typing import Any, Dict, Optional, List
from src.core.user import User, BaseUserRepository, UserInteraction

from .serializers import UserSerializer, InteractionSerializer, UserKnowledgeStateSerializer


class UserRepository(BaseUserRepository):
    """Repository for user data persistence"""
    
    def __init__(self, storage_path: str = "data/users/"):
        self.storage_path = storage_path
        self.cache: Dict[str, User] = {}
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "interactions"), exist_ok=True)
    
    def _get_user_path(self, user_id: str) -> str:
        """Get file path for a user"""
        return os.path.join(self.storage_path, f"{user_id}.json")
    
    def _get_interactions_path(self, user_id: str) -> str:
        """Get directory path for user interactions"""
        return os.path.join(self.storage_path, "interactions", user_id)
    
    def save_user(self, user: User) -> User:
        """Save user data"""
        filepath = self._get_user_path(user.user_id)
        
        with open(filepath, 'w') as f:
            json.dump(UserSerializer.serialize_user(user), f, indent=2)
        
        # Update cache
        self.cache[user.user_id] = user

        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            # Check cache first
            if user_id in self.cache:
                return self.cache[user_id]
            
            # Load from file if not in cache
            filepath = self._get_user_path(user_id)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                user = UserSerializer.deserialize_user(data)
                self.cache[user_id] = user
                return user
        except Exception as e:
            print(f"Error loading user {user_id}: {e}")
            return None
        
    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        filepath = self._get_user_path(user_id)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            # Remove from cache
            if user_id in self.cache:
                del self.cache[user_id]

            # Remove interactions
            interactions_path = self._get_interactions_path(user_id)
            if os.path.exists(interactions_path):
                import shutil
                shutil.rmtree(interactions_path)
            
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
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        filepath = self._get_user_path(user_id)
        return os.path.exists(filepath)
    
    def save_interaction(self, user_id: str, interaction: UserInteraction) -> bool:
        """Save user interaction"""
        # Ensure user exists
        user = self.get_user(user_id)
        if not user:
            # Create new user if doesn't exist
            user = User(user_id=user_id)
            self.save_user(user)
        
        # Add interaction to user
        user.interaction_history.append(interaction)
        
        # Also save interaction to separate file for analytics
        interactions_dir = self._get_interactions_path(user_id)
        os.makedirs(interactions_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        interaction_file = os.path.join(interactions_dir, f"{timestamp}.json")
        
        with open(interaction_file, 'w', encoding='utf-8') as f:
            json.dump(InteractionSerializer.serialize(interaction), f, indent=2)
        
        # Update user in storage
        self.save_user(user)
        
        return True
    
    def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Get user interactions"""
        user = self.get_user(user_id)
        if not user:
            return []
        
        return user.interaction_history[-limit:]
    
    def get_interactions_by_date(self, user_id: str, start_date: datetime, end_date: datetime) -> List[UserInteraction]:
        """Get interactions within date range"""
        interactions_dir = self._get_interactions_path(user_id)
        if not os.path.exists(interactions_dir):
            return []
        
        interactions = []
        for filename in os.listdir(interactions_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(interactions_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    interaction = InteractionSerializer.deserialize(data)
                    
                    # Check date range
                    if interaction.timestamp and start_date <= interaction.timestamp <= end_date:
                        interactions.append(interaction)
            except Exception as e:
                print(f"Error loading interaction {filepath}: {e}")
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min)
        
        return interactions
    
    def delete_interactions(self, user_id: str, before_date: datetime) -> int:
        """Delete interactions older than date"""
        interactions_dir = self._get_interactions_path(user_id)
        if not os.path.exists(interactions_dir):
            return 0
        
        deleted_count = 0
        for filename in os.listdir(interactions_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(interactions_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                    
                    if timestamp < before_date:
                        os.remove(filepath)
                        deleted_count += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        # Update user's in-memory interactions
        user = self.get_user(user_id)
        if user:
            user.interaction_history = [i for i in user.interaction_history 
                                       if i.timestamp and i.timestamp >= before_date]
            self.save_user(user)
        
        return deleted_count
    
    # Search methods
    def find_users_by_concept(self, concept_name: str, min_knowledge: float = 0.7) -> List[str]:
        """Find users who know a specific concept above threshold"""
        matching_users = []
        
        for user_id in self.list_users():
            user = self.get_user(user_id)
            if not user:
                continue
            
            for concept, state in user.knowledge_state.knowledge_states.items():
                if concept.name == concept_name and state.p_knowledge >= min_knowledge:
                    matching_users.append(user_id)
                    break
        
        return matching_users
    
    def find_users_by_interaction_count(self, min_interactions: int = 10) -> List[str]:
        """Find users with at least min_interactions"""
        matching_users = []
        
        for user_id in self.list_users():
            user = self.get_user(user_id)
            if user and len(user.interaction_history) >= min_interactions:
                matching_users.append(user_id)
        
        return matching_users
    
     # Statistics methods
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user"""
        user = self.get_user(user_id)
        if not user:
            return {}
        
        total_interactions = len(user.interaction_history)
        known_concepts = 0
        unknown_concepts = 0
        total_confidence = 0.0
        
        for state in user.knowledge_state.knowledge_states.values():
            if state.p_knowledge > 0.7:
                known_concepts += 1
            elif state.p_knowledge < 0.3:
                unknown_concepts += 1
            total_confidence += state.confidence
        
        avg_confidence = total_confidence / max(1, len(user.knowledge_state.knowledge_states))
        
        # Interaction types
        interaction_types = {}
        for interaction in user.interaction_history:
            interaction_types[interaction.interaction_type] = interaction_types.get(interaction.interaction_type, 0) + 1
        
        return {
            "user_id": user_id,
            "total_interactions": total_interactions,
            "known_concepts": known_concepts,
            "unknown_concepts": unknown_concepts,
            "total_concepts": len(user.knowledge_state.knowledge_states),
            "average_confidence": avg_confidence,
            "interaction_types": interaction_types,
            "last_active": user.interaction_history[-1].timestamp if user.interaction_history else None
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all users"""
        users = self.list_users()
        
        return {
            "total_users": len(users),
            "active_users": len([u for u in users if self.get_user(u) and len(self.get_user(u).interaction_history) > 0]),
            "users": [self.get_user_statistics(u) for u in users]
        }
    
    def export_user_data(self, user_id: str, export_path: str) -> bool:
        """Export all user data to a single file"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        data = {
            "user": self._serialize_user_full(user),
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting user data: {e}")
            return False
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self.cache.clear()
    
    def reload_user(self, user_id: str) -> Optional[User]:
        """Force reload user from disk"""
        if user_id in self.cache:
            del self.cache[user_id]
        return self.get_user(user_id)
    
    def _serialize_user_full(self, user: User) -> Dict:
        """Serialize user with full data"""
        return {
            "user_id": user.user_id,
            "knowledge_state":  UserKnowledgeStateSerializer.serialize(user.knowledge_state),
            "interaction_history": [InteractionSerializer.serialize(i) for i in user.interaction_history],
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    
    def import_user_data(self, import_path: str) -> Optional[User]:
        """Import user data from file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_data = data.get("user", {})
            user = UserSerializer.deserialize_user(user_data)
            
            # Save to repository
            self.save_user(user)
            
            return user
            
        except Exception as e:
            print(f"Error importing user data: {e}")
            return None
    