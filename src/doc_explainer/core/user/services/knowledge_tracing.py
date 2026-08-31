from typing import Dict, List, Optional
from datetime import datetime
from ...knowledge.models import Concept
from ..models import UserKnowledgeState, KnowledgeState,  UserInteraction

class BayesianKnowledgeTracer:
    """
    Implements Bayesian Knowledge Tracing (BKT) for user modeling
    
    BKT parameters for each concept:
    - p(L0): Initial probability of knowing
    - p(T): Probability of learning after opportunity
    - p(G): Probability of guessing correctly if unknown
    - p(S): Probability of slipping if known
    """
    
    def __init__(self, user_state: UserKnowledgeState):
        self.user_state = user_state
        # Pointer for easier access
        self.user_states = self.user_state.knowledge_states

    def update_from_interaction(self, concept_obj: Concept, response_data: Dict):
        """
        Update knowledge state based on user interaction
        
        response_data should contain:
        - correct: bool (whether response was correct)
        - time_spent: float (seconds spent)
        - explanation_depth: str ('beginner', 'intermediate', 'advanced')
        - asked_question: bool (whether user asked follow-up)
        """
        if concept_obj not in self.user_states:
            self.user_states[concept_obj] = KnowledgeState(
                concept=concept_obj,
                p_knowledge=0.1,
                p_learn=0.3,
                p_guess=0.2,
                p_slip=0.1,
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5
            )
        
        state = self.user_states[concept_obj]
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
                # Correct response
                state.n_correct += 1
                # P(L|correct) = (1 - p_slip) * p_knowledge / [(1 - p_slip) * p_knowledge + p_guess * (1 - p_knowledge)]
                numerator = (1 - state.p_slip) * state.p_knowledge
                denominator = numerator + state.p_guess * (1 - state.p_knowledge)
            else:
                # Incorrect response
                # P(L|incorrect) = p_slip * p_knowledge / [p_slip * p_knowledge + (1 - p_guess) * (1 - p_knowledge)]
                numerator = state.p_slip * state.p_knowledge
                denominator = numerator + (1 - state.p_guess) * (1 - state.p_knowledge)
            
            p_lt_given_action = numerator / denominator if denominator > 0 else 0
            
            # Account for learning transition
            state.p_knowledge = p_lt_given_action + (1 - p_lt_given_action) * state.p_learn

            # Clamp values
            state.p_knowledge = max(0.01, min(0.99, state.p_knowledge))

            # Update learning probability based on performance
            if correct and state.p_knowledge < 0.9:
                state.p_learn = min(0.8, state.p_learn + 0.05)
            elif not correct and state.p_knowledge < 0.5:
                state.p_learn = max(0.1, state.p_learn - 0.02)
        
        # Update based on time spent
        if time_spent > 60:  # More than 60 seconds
            state.p_knowledge *= 0.9
        elif time_spent < 10 and correct:  # Very quick correct response
            state.p_knowledge = min(0.999, state.p_knowledge * 1.1)
        
        # Update based on explanation depth
        if explanation_depth == 'beginner':
            state.p_knowledge *= 0.85
        elif explanation_depth == 'advanced':
            state.p_knowledge = min(1.0, state.p_knowledge * 1.15)
        
        # Update based on questions asked
        if asked_question:
            state.p_learn = min(0.8, state.p_learn + 0.1)
        
        # Update confidence based on number of observations
        state.confidence = min(0.95, 0.5 + (state.n_attempts * 0.1))
        
        # Ensure probabilities stay in valid range
        state.p_knowledge = max(0.01, min(0.99, state.p_knowledge))
        state.p_learn = max(0.05, min(0.9, state.p_learn))
        state.p_guess = max(0.05, min(0.5, state.p_guess))
        state.p_slip = max(0.01, min(0.3, state.p_slip))
        
        # Update the user_state confidence cache
        self.user_state.confidence[concept_obj.name] = state.p_knowledge
        
        return state

    def infer_knowledge_from_text(self, user_text: str, concepts: List[Concept]) -> Dict[str, float]:
        """
        Infer knowledge levels from user's free-form text
        Uses semantic similarity and keyword matching
        """
        from sentence_transformers import SentenceTransformer, util
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        user_embedding = embedder.encode(user_text, convert_to_tensor=True)
        
        knowledge_scores = {}
        
        for concept in concepts:
            # Get concept embedding
            if concept.embedding is not None:
                concept_embedding = concept.embedding
            else:
                concept_embedding = embedder.encode(concept.name, convert_to_tensor=True)
            
            # Semantic similarity
            semantic_sim = util.cos_sim(user_embedding, concept_embedding).item()
            
            # Keyword presence
            keyword_score = 1.0 if concept.name.lower() in user_text.lower() else 0.0
            
            # Combined score
            combined_score = 0.7 * semantic_sim + 0.3 * keyword_score
            
            # Update knowledge state if concept exists
            if concept in self.user_states:
                current_knowledge = self.user_states[concept].p_knowledge
                confidence = self.user_states[concept].confidence
                new_knowledge = (confidence * current_knowledge + combined_score) / (confidence + 1)
                self.user_states[concept].p_knowledge = new_knowledge
                self.user_states[concept].confidence = min(0.95, confidence + 0.1)
                
                # Update confidence cache
                self.user_state.confidence[concept.name] = new_knowledge
            
            knowledge_scores[concept.name] = combined_score
        
        return knowledge_scores

    def get_user_knowledge_state(self) -> UserKnowledgeState:
        """Get the current user knowledge state"""
        return self.user_state

    def initialize_user(self, concept_list: List[Concept]):
        """Initialize knowledge states for all concepts"""
        for concept in concept_list:
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
            self.user_state.confidence[concept.name] = 0.1

    def update_knowledge(self, user_response: Dict):
        """Update knowledge based on user response"""
        concept_name = user_response.get("concept") or user_response.get("text") or user_response.get("question")
        if not concept_name:
            return

        # Find concept in existing states
        concept = None
        for existing in self.user_states.keys():
            if existing.name == concept_name:
                concept = existing
                break
                
        if concept is None:
            # Create new concept if not found
            from src.core.knowledge.models.concept import Concept
            concept = Concept(name=concept_name)
            self.user_states[concept] = KnowledgeState(
                concept=concept,
                p_knowledge=0.2,
                p_learn=0.3,
                p_guess=0.2,
                p_slip=0.1,
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5
            )

        state = self.user_states[concept]
        state.n_attempts += 1
        state.last_interaction = datetime.now()
        state.p_knowledge = min(0.95, state.p_knowledge + 0.02)
        state.confidence = min(0.95, state.confidence + 0.02)
        
        # Update confidence cache
        self.user_state.confidence[concept.name] = state.p_knowledge

    def get_known_concepts(self, threshold: float = 0.7) -> List[Concept]:
        """Get concepts known above threshold"""
        return [
            concept for concept, state in self.user_states.items()
            if state.p_knowledge >= threshold and state.confidence > 0.6
        ]
    
    def get_unknown_concepts(self, threshold: float = 0.3) -> List[Concept]:
        """Get concepts unknown below threshold"""
        return [
            concept for concept, state in self.user_states.items()
            if state.p_knowledge <= threshold and state.n_attempts > 0
        ]