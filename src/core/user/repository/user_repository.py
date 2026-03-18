import json
import os
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

from ..models.user import User
from ..models.knowledge_state import UserKnowledgeState, KnowledgeState
from ..models.interaction import UserInteraction


class UserRepository:
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
        
        # Convert to serializable format
        user_data = {
            "user_id": user.user_id,
            "knowledge_state": self._serialize_knowledge_state(user.knowledge_state),
            "interaction_history": [self._serialize_interaction(i) for i in user.interaction_history],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        
        # Update cache
        self.cache[user.user_id] = user
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        # Check cache first
        if user_id in self.cache:
            return self.cache[user_id]
        
        # Load from file
        filepath = self._get_user_path(user_id)
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user = self._deserialize_user(data)
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
            if filename.endswith('.json') and filename != "interactions":
                users.append(filename.replace('.json', ''))
        
        return users
    
    def update_user(self, user: User) -> User:
        """Update existing user"""
        return self.save_user(user)
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        filepath = self._get_user_path(user_id)
        return os.path.exists(filepath)
    
    # Interaction methods
    
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
            json.dump(self._serialize_interaction(interaction), f, indent=2)
        
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
                    interaction = self._deserialize_interaction(data)
                    
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
    
    def get_user_statistics(self, user_id: str) -> Dict[str, any]:
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
    
    def get_all_statistics(self) -> Dict[str, any]:
        """Get statistics for all users"""
        users = self.list_users()
        
        return {
            "total_users": len(users),
            "active_users": len([u for u in users if self.get_user(u) and len(self.get_user(u).interaction_history) > 0]),
            "users": [self.get_user_statistics(u) for u in users]
        }
    
    # Serialization helpers
    
    def _serialize_knowledge_state(self, state: UserKnowledgeState) -> Dict:
        """Serialize knowledge state"""
        knowledge_states = {}
        for concept, ks in state.knowledge_states.items():
            knowledge_states[concept.name] = {
                "p_knowledge": ks.p_knowledge,
                "p_learn": ks.p_learn,
                "p_guess": ks.p_guess,
                "p_slip": ks.p_slip,
                "n_attempts": ks.n_attempts,
                "n_correct": ks.n_correct,
                "confidence": ks.confidence,
                "last_interaction": ks.last_interaction.isoformat() if ks.last_interaction else None
            }
        
        return {
            "knowledge_states": knowledge_states,
            "confidence": state.confidence,
            "exposure": state.exposure,
            "last_seen": state.last_seen
        }
    
    def _deserialize_knowledge_state(self, data: Dict) -> UserKnowledgeState:
        """Deserialize knowledge state"""
        from src.core.knowledge.models.concept import Concept
        
        state = UserKnowledgeState()
        state.confidence = data.get("confidence", {})
        state.exposure = data.get("exposure", {})
        state.last_seen = data.get("last_seen", {})
        
        # Note: Knowledge states require concept objects which need to be loaded separately
        # This will be populated when the user model is loaded with concepts
        
        return state
    
    def _serialize_interaction(self, interaction: UserInteraction) -> Dict:
        """Serialize interaction"""
        return {
            "interaction_id": interaction.interaction_id,
            "user_id": interaction.user_id,
            "interaction_type": interaction.interaction_type,
            "content": interaction.content,
            "context": interaction.context,
            "timestamp": interaction.timestamp.isoformat() if interaction.timestamp else None,
            "metadata": interaction.metadata
        }
    
    def _deserialize_interaction(self, data: Dict) -> UserInteraction:
        """Deserialize interaction"""
        from datetime import datetime
        
        interaction = UserInteraction(
            interaction_id=data.get("interaction_id", ""),
            user_id=data.get("user_id", ""),
            interaction_type=data.get("interaction_type", ""),
            content=data.get("content", {}),
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )
        
        timestamp = data.get("timestamp")
        if timestamp:
            interaction.timestamp = datetime.fromisoformat(timestamp)
        
        return interaction
    
    def _deserialize_user(self, data: Dict) -> User:
        """Deserialize user from dictionary"""
        from datetime import datetime
        
        user = User(
            user_id=data["user_id"],
            knowledge_state=self._deserialize_knowledge_state(data.get("knowledge_state", {})),
            interaction_history=[]
        )
        
        # Deserialize interactions
        for interaction_data in data.get("interaction_history", []):
            user.interaction_history.append(self._deserialize_interaction(interaction_data))
        
        return user
    
    # Import/Export methods
    
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
    
    def _serialize_user_full(self, user: User) -> Dict:
        """Serialize user with full data"""
        return {
            "user_id": user.user_id,
            "knowledge_state": self._serialize_knowledge_state(user.knowledge_state),
            "interaction_history": [self._serialize_interaction(i) for i in user.interaction_history],
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    
    def import_user_data(self, import_path: str) -> Optional[User]:
        """Import user data from file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_data = data.get("user", {})
            user = self._deserialize_user(user_data)
            
            # Save to repository
            self.save_user(user)
            
            return user
            
        except Exception as e:
            print(f"Error importing user data: {e}")
            return None
    
    # Cache management
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self.cache.clear()
    
    def reload_user(self, user_id: str) -> Optional[User]:
        """Force reload user from disk"""
        if user_id in self.cache:
            del self.cache[user_id]
        return self.get_user(user_id)