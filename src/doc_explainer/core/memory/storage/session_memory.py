from typing import Dict, List, Optional
from datetime import datetime
from .base import MemoryStorage
from ..base.interfaces import SessionMemoryInterface
from ..models.context import SessionContext


class SessionMemory(MemoryStorage, SessionMemoryInterface):
    """Stores session-level memory"""
    
    def __init__(self):
        super().__init__()
        self.session_data = SessionContext()
    
    def get_session_context(self) -> SessionContext:
        """Get current session context"""
        return self.session_data
    
    def update_session_context(self, **kwargs) -> bool:
        """Update session context with new information"""
        try:
            if 'interactions' in kwargs:
                for interaction in kwargs['interactions']:
                    self.session_data.add_interaction(interaction)
            
            if 'concepts' in kwargs:
                self.session_data.update_concepts(kwargs['concepts'])
            
            if 'preferences' in kwargs:
                self.session_data.update_preferences(kwargs['preferences'])
            
            return True
        except Exception as e:
            print(f"Error updating session context: {e}")
            return False
    
    def add_interaction(self, interaction: Dict) -> bool:
        """Add interaction to session"""
        self.session_data.add_interaction(interaction)
        return True
    
    def set_session_id(self, session_id: str):
        """Set session ID"""
        self.session_data.session_id = session_id
    
    def clear_session(self):
        """Clear current session"""
        self.session_data = SessionContext()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return self.session_data.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionMemory':
        """Create from dictionary"""
        memory = cls()
        memory.session_data = SessionContext.from_dict(data)
        return memory