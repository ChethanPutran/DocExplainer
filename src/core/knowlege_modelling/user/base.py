from typing import Dict, List
from dataclasses import dataclass,field
from datetime import datetime
import time
from  core.knowlege_modelling.graph.base import Concept

@dataclass
class UserInteraction:   
    """Stores user interaction data"""
    subject: str = ""
    level: str = ""
    mastery: float = 0.0
    last_seen: str = ""
    time_spent: float = 0.0
    quiz_response: str = ""
    questions_asked: List[str] = field(default_factory=list)
    explanation_depth_requested: str = ""
    source: str = "Chapter 3"

@dataclass
class KnowledgeState:
    """Individual concept knowledge state"""
    concept: Concept
    p_knowledge: float  # Probability of knowing (0-1)
    p_learn: float      # Learning rate
    p_guess: float      # Guess probability
    p_slip: float       # Slip probability
    n_attempts: int     # Number of attempts
    n_correct: int      # Number of correct responses
    last_interaction: datetime
    confidence: float   # Confidence in estimate


class UserKnowledgeState:
    """Stores overall user state"""
    def __init__(self):
        self.knowledge_states: Dict[Concept, KnowledgeState] = {}
        self.interaction_history: List[UserInteraction] = []
        self.confidence = {}      # cid -> [0,1]
        self.exposure = {}        # cid -> int
        self.last_seen = {}       # cid -> timestamp
    
    def to_dict(self) -> Dict:
        return {
            'knowledge_states': {
                concept.name: {
                    'p_knowledge': state.p_knowledge,
                    'p_learn': state.p_learn,
                    'p_guess': state.p_guess,
                    'p_slip': state.p_slip,
                    'n_attempts': state.n_attempts,
                    'n_correct': state.n_correct,
                    'last_interaction': state.last_interaction.isoformat(),
                    'confidence': state.confidence
                } for concept, state in self.knowledge_states.items()
            },
            'interaction_history': [
                # Using dataclasses.asdict would be cleaner here
                vars(interaction) for interaction in self.interaction_history
            ]
        }
    
    def update(self, cid: str, signal: float, alpha: float = 0.85):
        """
        
        Update user state for a concept based on new signal

        Each time a concept appears:

        cv(t)=α⋅cv(t−1)+(1−α)⋅sv(t)

        Where:
            sv(t): signal (read, answered question, lingered, skipped)
            α: memory decay

        This is basically Bayesian belief update / exponential smoothing.
        
        """
        prev = self.confidence.get(cid, 0.0)
        self.confidence[cid] = alpha * prev + (1 - alpha) * signal
        self.exposure[cid] = self.exposure.get(cid, 0) + 1
        self.last_seen[cid] = time.time()


class User:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.knowledge_state = UserKnowledgeState()
        self.interaction_history = []

    def record_interaction(self, interaction: UserInteraction):
        pass 