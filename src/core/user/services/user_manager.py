from typing import Dict, Optional
from src.core.user.models.user import User
from src.core.user.models.knowledge_state import UserKnowledgeState
from src.core.user.repository.user_repository import UserRepository
from src.core.user.services.knowledge_tracing import BayesianKnowledgeTracer

class UserManager:
    """Manages user operations and knowledge state"""
    
    def __init__(self, user_id: str, user_repository: UserRepository):
        self.user_id = user_id
        self.user_repository = user_repository
        self.user: User = self.user_repository.get_user(self.user_id)
        
        if not self.user:
            self.user = User(user_id=self.user_id, knowledge_state=UserKnowledgeState())
            self.user_repository.save_user(self.user)
        
        self.bkt = BayesianKnowledgeTracer(self.user.knowledge_state)
        
    def get_user(self) -> User:
        """Get the current user"""
        return self.user
    
    def update_user_knowledge(self, knowledge_update: Dict):
        """Update user knowledge based on interaction"""
        self.bkt.update_knowledge(knowledge_update)
        self.user_repository.save_user(self.user)

    def get_user_knowledge(self) -> UserKnowledgeState:
        """Get user knowledge state"""
        return self.bkt.get_user_knowledge_state()
    
    def user_confidence(self, concept_name: str) -> float:
        """Get user confidence for a concept"""
        return self.user.get_confidence(concept_name)
    
    def process_interaction(self, interaction_data: Dict):
        """Process a user interaction"""
        from ..models.interaction import UserInteraction
        
        interaction = UserInteraction(
            subject=interaction_data.get('subject', ''),
            level=interaction_data.get('level', ''),
            time_spent=interaction_data.get('time_spent', 0),
            quiz_response=interaction_data.get('quiz_response', ''),
            explanation_depth_requested=interaction_data.get('explanation_depth', ''),
            source=interaction_data.get('source', ''),
            correct=interaction_data.get('correct')
        )
        
        self.user.record_interaction(interaction)
        self.user_repository.save_user(self.user)
    
    def initialize_concepts(self, concepts: list):
        """Initialize knowledge states for concepts"""
        self.bkt.initialize_user(concepts)
        self.user_repository.save_user(self.user)