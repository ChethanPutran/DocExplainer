from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING, Any
from ..repository import BaseKnowledgeStore

if TYPE_CHECKING:
    from ...user import UserManager

class LearningPathGenerator:
    """Generates personalized learning paths"""
    
    def __init__(self, 
                 knowledge_store: BaseKnowledgeStore,
                 user_manager: UserManager):
        self.knowledge_store = knowledge_store
        self.user_manager = user_manager

    def generate_path(self, target_concept: str, max_depth: int = 3) -> List[Dict]:
        """Generate learning path to target concept"""
        concept_graph = self.knowledge_store.graph
        user_state = self.user_manager.get_user_knowledge()
        
        if target_concept not in concept_graph.graph:
            return []
        
        # Get prerequisites
        prerequisites = concept_graph.get_prerequisites(
            target_concept, 
            user_state.get_confidence(target_concept)
        )
        
        # Build learning path
        path = []
        seen = set()
        
        for prereq in prerequisites:
            concept_name = prereq["concept"]
            if concept_name in seen:
                continue
                
            seen.add(concept_name)
            
            # Get user knowledge for this concept
            confidence = user_state.confidence.get(concept_name, 0.0)
            
            path.append({
                "concept": concept_name,
                "depth": prereq["depth"],
                "relation": prereq["relation"],
                "importance": prereq["importance"],
                "user_confidence": confidence,
                "priority": (1.0 - confidence) * prereq["importance"]
            })
        
        # Sort by priority
        path.sort(key=lambda x: x["priority"], reverse=True)
        
        return path[:max_depth * 3]  # Limit results

    def get_next_concepts(self, current_concepts: List[str], limit: int = 5) -> List[Dict]:
        """Get recommended next concepts to learn"""
        concept_graph = self.knowledge_store.graph
        user_state = self.user_manager.get_user_knowledge()
        
        candidates = []
        
        for concept_name in current_concepts:
            if concept_name not in concept_graph.graph:
                continue
                
            # Get outgoing edges
            for _, target, data in concept_graph.graph.out_edges(concept_name, data=True):
                if target in current_concepts:
                    continue
                    
                rel_wrapper = data.get('relationship')
                if not rel_wrapper:
                    continue
                
                # Check if user already knows this
                if user_state.confidence.get(target, 0.0) > 0.7:
                    continue
                
                candidates.append({
                    "concept": target,
                    "from_concept": concept_name,
                    "relation": rel_wrapper.relationship.relation,
                    "strength": rel_wrapper.relationship.strength
                })
        
        # Score and sort candidates
        for candidate in candidates:
            confidence = user_state.confidence.get(candidate["concept"], 0.0)
            candidate["score"] = candidate["strength"] * (1.0 - confidence)
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates[:limit]