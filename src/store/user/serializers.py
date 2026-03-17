from typing import Dict, Any
from src.core.user.models.user import User
from src.core.user.models.knowledge_state import UserKnowledgeState, KnowledgeState
from src.core.user.models.interaction import UserInteraction
from src.core.knowledge.models.concept import Concept


class UserSerializer:
    """Serializer for User objects"""
    
    @staticmethod
    def serialize_user(user: User) -> Dict[str, Any]:
        """Serialize user to dictionary"""
        return {
            'user_id': user.user_id,
            'knowledge_state': UserKnowledgeStateSerializer.serialize(user.knowledge_state),
            'interaction_history': [
                InteractionSerializer.serialize(i) for i in user.interaction_history
            ]
        }
    
    @staticmethod
    def deserialize_user(data: Dict[str, Any]) -> User:
        """Deserialize user from dictionary"""
        knowledge_state = UserKnowledgeStateSerializer.deserialize(
            data.get('knowledge_state', {})
        )
        
        interaction_history = [
            InteractionSerializer.deserialize(i) 
            for i in data.get('interaction_history', [])
        ]
        
        return User(
            user_id=data['user_id'],
            knowledge_state=knowledge_state,
            interaction_history=interaction_history
        )


class UserKnowledgeStateSerializer:
    """Serializer for UserKnowledgeState objects"""
    
    @staticmethod
    def serialize(state: UserKnowledgeState) -> Dict[str, Any]:
        """Serialize user knowledge state"""
        return {
            'confidence': state.confidence,
            'exposure': state.exposure,
            'last_seen': state.last_seen,
            # Note: knowledge_states requires concept objects which are stored separately
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> UserKnowledgeState:
        """Deserialize user knowledge state"""
        state = UserKnowledgeState()
        state.confidence = data.get('confidence', {})
        state.exposure = data.get('exposure', {})
        state.last_seen = data.get('last_seen', {})
        return state


class InteractionSerializer:
    """Serializer for UserInteraction objects"""
    
    @staticmethod
    def serialize(interaction: UserInteraction) -> Dict[str, Any]:
        """Serialize interaction"""
        return interaction.to_dict()
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> UserInteraction:
        """Deserialize interaction"""
        return UserInteraction.from_dict(data)