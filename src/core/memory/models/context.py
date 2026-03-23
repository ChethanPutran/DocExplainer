from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional,  TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.core.knowledge import ConceptGraph
    from src.core.user import UserKnowledgeState    

@dataclass
class SessionContext:
    """Holds session context information"""
    interactions: List[Dict] = field(default_factory=list)
    concepts: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    start_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now()
    
    def update_concepts(self, new_concepts: Dict):
        """Update concepts in session context"""
        self.concepts.update(new_concepts)
    
    def update_preferences(self, new_preferences: Dict):
        """Update user preferences in session context"""
        self.preferences.update(new_preferences)
    
    def add_interaction(self, interaction: Dict):
        """Add a user interaction to session context"""
        if 'timestamp' not in interaction:
            from datetime import datetime
            interaction['timestamp'] = datetime.now().isoformat()
        self.interactions.append(interaction)
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get most recent interactions"""
        return self.interactions[-limit:]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'interactions': self.interactions,
            'concepts': self.concepts,
            'preferences': self.preferences,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionContext':
        """Create from dictionary"""
        context = cls(
            interactions=data.get('interactions', []),
            concepts=data.get('concepts', {}),
            preferences=data.get('preferences', {}),
            session_id=data.get('session_id', '')
        )
        if data.get('start_time'):
            from datetime import datetime
            context.start_time = datetime.fromisoformat(data['start_time'])
        return context


@dataclass
class Context:
    """Holds comprehensive context for explanations"""
    user_knowledge: UserKnowledgeState
    session_context: SessionContext
    document_context: Any  # Document context
    concept_graph: ConceptGraph
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'user_knowledge': self.user_knowledge.to_dict() if self.user_knowledge else {},
            'session_context': self.session_context.to_dict(),
            'document_context': str(self.document_context),  # Simplified
            'concept_graph': 'ConceptGraph'  # Placeholder
        }