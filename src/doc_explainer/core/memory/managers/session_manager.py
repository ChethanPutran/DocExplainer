from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional
from ..storage.session_memory import SessionMemory
from ..chains.session_chain import SessionChain
from ..models.context import SessionContext


class SessionManager:
    """Manages session-level interactions and context"""
    
    def __init__(self):
        self.session_memory = SessionMemory()
        self.session_chain = SessionChain()
    
    def get_session_context(self) -> SessionContext:
        """Retrieve the current session context"""
        return self.session_memory.get_session_context()
    
    def update_session_context(self, interactions=None, concepts=None, preferences=None) -> bool:
        """Update the session context with new information"""
        updates = {}
        if interactions is not None:
            updates['interactions'] = interactions
        if concepts is not None:
            updates['concepts'] = concepts
        if preferences is not None:
            updates['preferences'] = preferences
        
        return self.session_memory.update_session_context(**updates)
    
    def handle_interaction(self, name: str, interaction: Any) -> bool:
        """Handle a user interaction and update session context"""
        if not isinstance(interaction, dict):
            interaction = {
                "text": str(interaction),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            interaction = dict(interaction)
            interaction.setdefault(
                "timestamp",
                datetime.now().isoformat(),
            )

        # Add to chain
        self.session_chain.add_interaction(name, interaction)
        
        # Add to memory
        return self.session_memory.add_interaction({
            "name": name,
            "data": interaction
        })
    
    def get_recent_interactions(self, limit: int = 10) -> list:
        """Get recent interactions"""
        return self.session_memory.session_data.get_recent_interactions(limit)
    
    def get_session_chain(self) -> 'SessionChain':
        """Get the session chain"""
        return self.session_chain
    
    def clear_session(self):
        """Clear current session"""
        self.session_memory.clear_session()
        self.session_chain.clear_graph()
    
    def set_session_id(self, session_id: str):
        """Set session ID"""
        self.session_memory.set_session_id(session_id)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "session_memory": self.session_memory.to_dict(),
            "session_chain": self.session_chain.get_graph()
        }