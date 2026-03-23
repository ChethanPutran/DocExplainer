from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
from src.core.knowledge import Concept

@dataclass
class KnowledgeState:
    """Individual concept knowledge state"""
    concept: Concept
    p_knowledge: float = 0.1  # Probability of knowing (0-1)
    p_learn: float = 0.3      # Learning rate
    p_guess: float = 0.2      # Guess probability
    p_slip: float = 0.1       # Slip probability
    n_attempts: int = 0       # Number of attempts
    n_correct: int = 0        # Number of correct responses
    last_interaction: Optional[datetime] = None
    confidence: float = 0.5    # Confidence in estimate
    
    def __post_init__(self):
        if self.last_interaction is None:
            self.last_interaction = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'concept_name': self.concept.name,
            'p_knowledge': self.p_knowledge,
            'p_learn': self.p_learn,
            'p_guess': self.p_guess,
            'p_slip': self.p_slip,
            'n_attempts': self.n_attempts,
            'n_correct': self.n_correct,
            'last_interaction': self.last_interaction.isoformat() if self.last_interaction else None,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict, concept: Concept) -> 'KnowledgeState':
        """Create from dictionary"""
        state = cls(concept=concept)
        state.p_knowledge = data.get('p_knowledge', 0.1)
        state.p_learn = data.get('p_learn', 0.3)
        state.p_guess = data.get('p_guess', 0.2)
        state.p_slip = data.get('p_slip', 0.1)
        state.n_attempts = data.get('n_attempts', 0)
        state.n_correct = data.get('n_correct', 0)
        state.confidence = data.get('confidence', 0.5)
        
        last_interaction = data.get('last_interaction')
        if last_interaction:
            state.last_interaction = datetime.fromisoformat(last_interaction)
        
        return state


class UserKnowledgeState:
    """Stores overall user knowledge state"""
    
    def __init__(self):
        self.knowledge_states: Dict[Concept, KnowledgeState] = {}
        self.interaction_history: List = []
        self.confidence: Dict[str, float] = {}      # concept_name -> [0,1]
        self.exposure: Dict[str, int] = {}          # concept_name -> count
        self.last_seen: Dict[str, float] = {}       # concept_name -> timestamp
    
    def update(self, concept_name: str, signal: float, alpha: float = 0.85):
        """
        Update user state for a concept based on new signal
        
        cv(t) = α * cv(t-1) + (1-α) * sv(t)
        
        Where:
            sv(t): signal (read, answered question, lingered, skipped)
            α: memory decay
        """
        prev = self.confidence.get(concept_name, 0.0)
        self.confidence[concept_name] = alpha * prev + (1 - alpha) * signal
        self.exposure[concept_name] = self.exposure.get(concept_name, 0) + 1
        self.last_seen[concept_name] = time.time()

    def get_confidence(self, concept_name: str) -> float:
        """Get current confidence for a concept"""
        return self.confidence.get(concept_name, 0.0)
    
    def get_knowledge_state(self, concept: Concept) -> Optional[KnowledgeState]:
        """Get knowledge state for a concept"""
        return self.knowledge_states.get(concept)
    
    def update_from_interaction(self, interaction):
        """Update state based on user interaction"""
        # This will be implemented with BKT
        pass
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'knowledge_states': {
                concept.name: state.to_dict() 
                for concept, state in self.knowledge_states.items()
            },
            'confidence': self.confidence,
            'exposure': self.exposure,
            'last_seen': self.last_seen
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserKnowledgeState':
        """Create from dictionary"""
        state = cls()
        state.confidence = data.get('confidence', {})
        state.exposure = data.get('exposure', {})
        state.last_seen = data.get('last_seen', {})
        # Note: knowledge_states requires concept objects which need to be loaded separately
        return state