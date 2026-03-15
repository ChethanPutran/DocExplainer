from typing import Dict, List
import pickle
from datetime import datetime
from core.knowlege_modelling.graph.base import Concept
from core.knowlege_modelling.user.base import UserKnowledgeState, KnowledgeState

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
        self.user_state = UserKnowledgeState()
        # Pointer for easier access
        self.user_states = self.user_state.knowledge_states

    def update_from_interaction(self, concept_obj: Concept, response_data: Dict):
        """
        Update knowledge state based on user interaction
        Implements the BKT posterior update:
        P(L_n-1 | Result) -> P(L_n) = P(L_n-1|Result) + (1 - P(L_n-1|Result)) * P(T)
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
                # 1. Calculate P(L | Action) - Posterior
                state.n_correct += 1
                # If correct and known: (1 - p_slip) * p_knowledge
                # If correct and unknown: p_guess * (1 - p_knowledge)
                numerator = (1 - state.p_slip) * state.p_knowledge
                denominator = numerator + state.p_guess * (1 - state.p_knowledge)
               
            else:
                # If incorrect and known: p_slip * p_knowledge
                # If incorrect and unknown: (1 - p_guess) * (1 - p_knowledge)
                numerator = state.p_slip * state.p_knowledge
                denominator = numerator + (1 - state.p_guess) * (1 - state.p_knowledge)
                
      
            p_lt_given_action = numerator / denominator if denominator > 0 else 0
            
            # 2. Account for transition (Learning)
            state.p_knowledge = p_lt_given_action + (1 - p_lt_given_action) * state.p_learn

            # Clamp values
            state.p_knowledge = max(0.01, min(0.99, state.p_knowledge))

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
                state.p_knowledge = min(0.999, state.p_knowledge * 1.1)
        
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
            # Get concept embedding (could be pre-computed)
            concept_embedding = embedder.encode(concept, convert_to_tensor=True)
            
            # Semantic similarity
            semantic_sim = util.cos_sim(user_embedding, concept_embedding).item()
            
            # Keyword presence
            keyword_score = 1.0 if concept.lower() in user_text.lower() else 0.0
            
            # Combined score
            combined_score = 0.7 * semantic_sim + 0.3 * keyword_score

            #  # Combine similarity with keyword matching
            # keyword_bonus = 0.3 if concept.name.lower() in user_text.lower() else 0.0
            # total_score = min(1.0, similarity + keyword_bonus)
            
            # Update knowledge state if concept exists
            if concept in self.user_states:
                current_knowledge = self.user_states[concept].p_knowledge
                # Weighted update: trust new evidence more if we have little data
                confidence = self.user_states[concept].confidence
                new_knowledge = (confidence * current_knowledge + combined_score) / (confidence + 1)
                self.user_states[concept].p_knowledge = new_knowledge
                self.user_states[concept].confidence = min(0.95, confidence + 0.1)

                #    # Update existing state using exponential smoothing
                # alpha = 0.2 # Trust text inference less than direct quiz results
                # self.user_states[concept].p_knowledge = (
                #     (1 - alpha) * self.user_states[concept].p_knowledge + alpha * total_score
                # )
            
            knowledge_scores[concept] = combined_score
        
        return knowledge_scores

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
        concept_name = user_response.get("concept") or user_response.get("text") or user_response.get("question")
        if not concept_name:
            return

        concept = None
        for existing in self.user_state.knowledge_states.keys():
            if existing.name == str(concept_name):
                concept = existing
                break
        if concept is None:
            concept = Concept(name=str(concept_name), description=str(concept_name))
            self.user_state.knowledge_states[concept] = KnowledgeState(
                concept=concept,
                p_knowledge=0.2,
                p_learn=0.3,
                p_guess=0.2,
                p_slip=0.1,
                n_attempts=0,
                n_correct=0,
                last_interaction=datetime.now(),
                confidence=0.5,
            )

        state = self.user_state.knowledge_states[concept]
        state.n_attempts += 1
        state.last_interaction = datetime.now()
        state.p_knowledge = min(0.95, state.p_knowledge + 0.02)
        state.confidence = min(0.95, state.confidence + 0.02)

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
    
    def recommend_learning_path(self, target_concept: Concept, concept_graph) -> List[Dict]:
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

