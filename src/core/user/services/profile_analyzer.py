from typing import Dict, List, Any
from src.core.knowledge import ConceptGraph

from ..models import User,  KnowledgeState

class ProfileAnalyzer:
    """Analyzes user profile and provides insights"""
    
    def __init__(self, user: User):
        self.user = user
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get comprehensive user profile summary"""
        user_state = self.user.knowledge_state
        knowledge_states = user_state.knowledge_states
        
        known_concepts = []
        unknown_concepts = []
        learning_concepts = []
        
        for concept, state in knowledge_states.items():
            profile = {
                'concept': concept.name,
                'knowledge': state.p_knowledge,
                'confidence': state.confidence,
                'attempts': state.n_attempts,
                'correct': state.n_correct
            }
            
            if state.p_knowledge > 0.7 and state.confidence > 0.6:
                known_concepts.append(profile)
            elif state.p_knowledge < 0.3 and state.n_attempts > 0:
                unknown_concepts.append(profile)
            else:
                learning_concepts.append(profile)
        
        # Sort by knowledge level
        known_concepts.sort(key=lambda x: x['knowledge'], reverse=True)
        unknown_concepts.sort(key=lambda x: x['knowledge'])
        learning_concepts.sort(key=lambda x: x['knowledge'], reverse=True)
        
        # Calculate overall metrics
        total_concepts = len(knowledge_states)
        if total_concepts > 0:
            avg_knowledge = sum(s.p_knowledge for s in knowledge_states.values()) / total_concepts
            avg_confidence = sum(s.confidence for s in knowledge_states.values()) / total_concepts
            total_attempts = sum(s.n_attempts for s in knowledge_states.values())
            total_correct = sum(s.n_correct for s in knowledge_states.values())
            accuracy = total_correct / total_attempts if total_attempts > 0 else 0
        else:
            avg_knowledge = 0
            avg_confidence = 0
            total_attempts = 0
            accuracy = 0
        
        return {
            'known_concepts': known_concepts[:10],
            'unknown_concepts': unknown_concepts[:10],
            'learning_concepts': learning_concepts[:10],
            'metrics': {
                'total_concepts_tracked': total_concepts,
                'average_knowledge': round(avg_knowledge, 3),
                'average_confidence': round(avg_confidence, 3),
                'total_interactions': total_attempts,
                'accuracy': round(accuracy, 3)
            }
        }
    
    def get_strengths_and_weaknesses(self) -> Dict[str, List[str]]:
        """Identify user strengths and weaknesses"""
        user_state = self.user.knowledge_state
        knowledge_states = user_state.knowledge_states
        
        strengths = []
        weaknesses = []
        
        for concept, state in knowledge_states.items():
            if state.p_knowledge > 0.8 and state.confidence > 0.7:
                strengths.append(concept.name)
            elif state.p_knowledge < 0.3 and state.n_attempts > 2:
                weaknesses.append(concept.name)
        
        return {
            'strengths': strengths[:10],
            'weaknesses': weaknesses[:10]
        }
    
    def get_learning_progress(self, days: int = 30) -> Dict[str, Any]:
        """Analyze learning progress over time"""
        # This would require storing historical data
        # For now, return placeholder
        return {
            'concepts_learned': 0,
            'average_progress': 0,
            'time_spent': 0
        }
    
    def recommend_focus_areas(self, concept_graph: ConceptGraph, limit: int = 5) -> List[Dict]:
        """Recommend concepts to focus on"""
        user_state = self.user.knowledge_state
        knowledge_states = user_state.knowledge_states
        
        focus_areas = []
        
        for concept, state in knowledge_states.items():
            # Concepts in the "learning zone" (40-70% knowledge)
            if 0.4 <= state.p_knowledge <= 0.7:
                # Check if concept has dependencies
                if concept_graph.has_concept(concept.name):
                    deps = concept_graph.get_dependencies(concept)
                    if deps:
                        # Check if dependencies are known
                        dep_known = all(
                            user_state.get_confidence(dep.name) > 0.7 
                            for dep in deps
                        )
                        if dep_known:
                            focus_areas.append({
                                'concept': concept.name,
                                'current_knowledge': state.p_knowledge,
                                'confidence': state.confidence,
                                'priority': (0.7 - state.p_knowledge) * state.confidence
                            })
        
        focus_areas.sort(key=lambda x: x['priority'], reverse=True)
        return focus_areas[:limit]