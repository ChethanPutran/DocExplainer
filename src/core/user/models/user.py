from typing import Dict, Optional
from .knowledge_state import UserKnowledgeState
from .interaction import UserInteraction

class User:
    """Represents a user in the system"""
    
    def __init__(self, 
                 user_id: str,
                 knowledge_state: Optional[UserKnowledgeState] = None,
                 interaction_history: Optional[list] = None):
        self.user_id = user_id
        self.knowledge_state = knowledge_state if knowledge_state is not None else UserKnowledgeState()
        self.interaction_history = interaction_history if interaction_history is not None else []

    def record_interaction(self, interaction: UserInteraction):
        """Record a user interaction"""
        self.interaction_history.append(interaction)
        # Update knowledge state based on interaction
        self.knowledge_state.update_from_interaction(interaction)

    def get_confidence(self, concept_name: str) -> float:
        """Get confidence for a concept"""
        return self.knowledge_state.get_confidence(concept_name)

    def to_dict(self) -> Dict:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'knowledge_state': self.knowledge_state.to_dict(),
            'interaction_history': [interaction.to_dict() for interaction in self.interaction_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """Create user from dictionary"""
        user = cls(
            user_id=data['user_id'],
            knowledge_state=UserKnowledgeState.from_dict(data.get('knowledge_state', {})),
            interaction_history=[
                UserInteraction.from_dict(item) for item in data.get('interaction_history', [])
            ]
        )
        return user