from typing import Dict, List
from dataclasses import dataclass
import pickle
from datetime import datetime
from  src.core.knowlege_modelling.base import Concept

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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
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
                {
                    'subject': interaction.subject,
                    'level': interaction.level,
                    'mastery': interaction.mastery,
                    'last_seen': interaction.last_seen,
                    'time_spent': interaction.time_spent,
                    'quiz_response': interaction.quiz_response,
                    'questions_asked': interaction.questions_asked,
                    'explanation_depth_requested': interaction.explanation_depth_requested,
                    'source': interaction.source
                } for interaction in self.interaction_history
            ]
        }
    
class UserInteraction:   
    """Stores user interaction data"""
    subject: str = ""
    level: str = ""
    mastery: float = 0.0
    last_seen: str = ""
    time_spent: str = ""
    quiz_response: str = ""
    questions_asked: List[str] | None = None
    explanation_depth_requested: str = ""
    source: str = "Chapter 3"

import time

class UserState:
    def __init__(self):
        self.confidence = {}      # cid -> [0,1]
        self.exposure = {}        # cid -> int
        self.last_seen = {}       # cid -> timestamp

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


class BayesianKnowledgeTracer:
    """
    Implements Bayesian Knowledge Tracing (BKT) for user modeling
    
    BKT parameters for each concept:
    - p(L0): Initial probability of knowing
    - p(T): Probability of learning after opportunity
    - p(G): Probability of guessing correctly if unknown
    - p(S): Probability of slipping if known
    """
    
    def __init__(self):
        self.user_state: UserKnowledgeState = UserKnowledgeState()
        self.concept_graph = None

    def get_user_knowledge_state(self) -> UserKnowledgeState:
        """Retrieve knowledge state for a concept"""
        return self.user_state

    def initialize_user(self, concept_list: List[Concept]):
        """Initialize knowledge states for all concepts"""
        for concept in concept_list:
            # Initialize with reasonable priors
            self.user_state.knowledge_states[concept] = KnowledgeState(
                concept=concept,
                p_knowledge=0.1,  # Assume unknown initially
                p_learn=0.3,      # Moderate learning rate
                p_guess=0.2,      # Low guess probability (for complex concepts)
                p_slip=0.1,       # Low slip probability
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5    # Low initial confidence
            )

    def update_knowledge(self, user_response:Dict):
        pass

    def update_from_interaction(self, concept: str, response_data: Dict):
        """
        Update knowledge state based on user interaction
        
        response_data should contain:
        - correct: bool (whether response was correct)
        - time_spent: float (seconds spent)
        - explanation_depth: str ('beginner', 'intermediate', 'advanced')
        - asked_question: bool (whether user asked follow-up)
        """
        if concept not in self.user_states:
            self.user_states[concept] = KnowledgeState(
                concept=concept,
                p_knowledge=0.1,
                p_learn=0.3,
                p_guess=0.2,
                p_slip=0.1,
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5
            )
        
        state = self.user_states[concept]
        state.last_interaction = datetime.now()
        state.n_attempts += 1
        
        # Extract interaction features
        correct = response_data.get('correct', None)
        time_spent = response_data.get('time_spent', 0)
        explanation_depth = response_data.get('explanation_depth', 'intermediate')
        asked_question = response_data.get('asked_question', False)
        
        # Bayesian update if we have correctness information
        if correct is not None:
            if correct:
                state.n_correct += 1
                # If correct and known: (1 - p_slip) * p_knowledge
                # If correct and unknown: p_guess * (1 - p_knowledge)
                numerator = (1 - state.p_slip) * state.p_knowledge
                denominator = numerator + state.p_guess * (1 - state.p_knowledge)
                if denominator > 0:
                    state.p_knowledge = numerator / denominator
            else:
                # If incorrect and known: p_slip * p_knowledge
                # If incorrect and unknown: (1 - p_guess) * (1 - p_knowledge)
                numerator = state.p_slip * state.p_knowledge
                denominator = numerator + (1 - state.p_guess) * (1 - state.p_knowledge)
                if denominator > 0:
                    state.p_knowledge = numerator / denominator
            
            # Update learning probability based on performance
            if correct and state.p_knowledge < 0.9:
                # Successful interaction increases learning rate slightly
                state.p_learn = min(0.8, state.p_learn + 0.05)
            elif not correct and state.p_knowledge < 0.5:
                # Difficulty might indicate need for different approach
                state.p_learn = max(0.1, state.p_learn - 0.02)
        
        # Update based on time spent (longer time might indicate difficulty)
        if time_spent > 60:  # More than 60 seconds
            state.p_knowledge *= 0.9  # Slight decrease
        elif time_spent < 10:  # Very quick response
            if correct:
                state.p_knowledge = min(1.0, state.p_knowledge * 1.1)
        
        # Update based on explanation depth requested
        if explanation_depth == 'beginner':
            # Requesting beginner explanation suggests lower knowledge
            state.p_knowledge *= 0.85
        elif explanation_depth == 'advanced':
            # Requesting advanced explanation suggests higher knowledge
            state.p_knowledge = min(1.0, state.p_knowledge * 1.15)
        
        # Update based on questions asked
        if asked_question:
            # Asking questions is good! Shows engagement
            state.p_learn = min(0.8, state.p_learn + 0.1)
        
        # Update confidence based on number of observations
        state.confidence = min(0.95, 0.5 + (state.n_attempts * 0.1))
        
        # Ensure probabilities stay in valid range
        state.p_knowledge = max(0.01, min(0.99, state.p_knowledge))
        state.p_learn = max(0.05, min(0.9, state.p_learn))
        state.p_guess = max(0.05, min(0.5, state.p_guess))
        state.p_slip = max(0.01, min(0.3, state.p_slip))
        
        return state
    
    def infer_knowledge_from_text(self, user_text: str, concepts: List[str]) -> Dict[str, float]:
        """
        Infer knowledge levels from user's free-form text
        Uses semantic similarity and keyword matching
        """
        from sentence_transformers import SentenceTransformer, util
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        user_embedding = embedder.encode(user_text, convert_to_tensor=True)
        
        knowledge_scores = {}
        
        for concept in concepts:
            # Get concept embedding (could be pre-computed)
            concept_embedding = embedder.encode(concept, convert_to_tensor=True)
            
            # Semantic similarity
            semantic_sim = util.cos_sim(user_embedding, concept_embedding).item()
            
            # Keyword presence
            keyword_score = 1.0 if concept.lower() in user_text.lower() else 0.0
            
            # Combined score
            combined_score = 0.7 * semantic_sim + 0.3 * keyword_score
            
            # Update knowledge state if concept exists
            if concept in self.user_states:
                current_knowledge = self.user_states[concept].p_knowledge
                # Weighted update: trust new evidence more if we have little data
                confidence = self.user_states[concept].confidence
                new_knowledge = (confidence * current_knowledge + combined_score) / (confidence + 1)
                self.user_states[concept].p_knowledge = new_knowledge
                self.user_states[concept].confidence = min(0.95, confidence + 0.1)
            
            knowledge_scores[concept] = combined_score
        
        return knowledge_scores
    
    def get_user_profile(self) -> Dict:
        """Get comprehensive user profile"""
        known_concepts = []
        unknown_concepts = []
        learning_concepts = []
        
        for concept, state in self.user_states.items():
            if state.p_knowledge > 0.7 and state.confidence > 0.6:
                known_concepts.append({
                    'concept': concept,
                    'knowledge': state.p_knowledge,
                    'confidence': state.confidence,
                    'attempts': state.n_attempts
                })
            elif state.p_knowledge < 0.3 and state.n_attempts > 0:
                unknown_concepts.append({
                    'concept': concept,
                    'knowledge': state.p_knowledge,
                    'confidence': state.confidence,
                    'attempts': state.n_attempts
                })
            else:
                learning_concepts.append({
                    'concept': concept,
                    'knowledge': state.p_knowledge,
                    'confidence': state.confidence,
                    'attempts': state.n_attempts
                })
        
        # Sort by knowledge level
        known_concepts.sort(key=lambda x: x['knowledge'], reverse=True)
        unknown_concepts.sort(key=lambda x: x['knowledge'])
        learning_concepts.sort(key=lambda x: x['knowledge'], reverse=True)
        
        # Calculate overall metrics
        total_concepts = len(self.user_states)
        if total_concepts > 0:
            avg_knowledge = sum(s.p_knowledge for s in self.user_states.values()) / total_concepts
            learning_rate = sum(s.p_learn for s in self.user_states.values()) / total_concepts
        else:
            avg_knowledge = 0
            learning_rate = 0
        
        return {
            'known_concepts': known_concepts[:10],  # Top 10 known
            'unknown_concepts': unknown_concepts[:10],  # Top 10 unknown
            'learning_concepts': learning_concepts[:10],
            'metrics': {
                'total_concepts_tracked': total_concepts,
                'average_knowledge': avg_knowledge,
                'average_learning_rate': learning_rate,
                'total_interactions': sum(s.n_attempts for s in self.user_states.values())
            }
        }
    
    def recommend_learning_path(self, target_concept: str, concept_graph) -> List[Dict]:
        """
        Recommend learning path to target concept based on knowledge gaps
        """
        if target_concept not in self.user_states:
            # Initialize if new concept
            self.user_states[target_concept] = KnowledgeState(
                concept=target_concept,
                p_knowledge=0.1,
                p_learn=0.3,
                p_guess=0.2,
                p_slip=0.1,
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5
            )
        
        # Find prerequisite chain
        prerequisites = concept_graph.find_prerequisites(
            target_concept, 
            set(self.get_known_concepts(threshold=0.7))
        )
        
        # Sort by user knowledge (unknown first)
        learning_path = []
        for prereq in prerequisites:
            concept = prereq['concept']
            if concept in self.user_states:
                knowledge = self.user_states[concept].p_knowledge
            else:
                knowledge = 0.1
            
            learning_path.append({
                'concept': concept,
                'knowledge_gap': 1.0 - knowledge,
                'importance': prereq['importance'],
                'relation': prereq['relation'],
                'current_knowledge': knowledge,
                'priority': (1.0 - knowledge) * prereq['importance']
            })
        
        # Sort by priority (high knowledge gap × high importance)
        learning_path.sort(key=lambda x: x['priority'], reverse=True)
        
        return learning_path
    
    def get_known_concepts(self, threshold: float = 0.7) -> List[str]:
        """Get concepts known above threshold"""
        return [
            concept for concept, state in self.user_states.items()
            if state.p_knowledge >= threshold and state.confidence > 0.6
        ]
    
    def get_unknown_concepts(self, threshold: float = 0.3) -> List[str]:
        """Get concepts unknown below threshold"""
        return [
            concept for concept, state in self.user_states.items()
            if state.p_knowledge <= threshold and state.n_attempts > 0
        ]
    
    def save_user_model(self, filepath: str):
        """Save user model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'user_states': self.user_states,
                'timestamp': datetime.now()
            }, f)
    
    def load_user_model(self, filepath: str):
        """Load user model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.user_states = data['user_states']