from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING


from ..repository import BaseKnowledgeStore

if TYPE_CHECKING:
    from ...user import UserManager

class RecommendationService:
    """Provides concept recommendations based on user state"""
    
    def __init__(self, 
                 knowledge_store: BaseKnowledgeStore,
                 user_manager: UserManager):
        self.knowledge_store = knowledge_store
        self.user_manager = user_manager

    def recommend_related_concepts(self, concept_name: str, limit: int = 5) -> List[Dict]:
        """Recommend concepts related to a given concept"""
        concept_graph = self.knowledge_store.graph
        user_state = self.user_manager.get_user_knowledge()
        
        if concept_name not in concept_graph.graph:
            return []
        
        recommendations = []
        
        # Get direct relationships
        for _, target, data in concept_graph.graph.out_edges(concept_name, data=True):
            rel_wrapper = data.get('relationship')
            if not rel_wrapper:
                continue
            
            # Skip if user already knows it well
            if user_state.confidence.get(target, 0.0) > 0.8:
                continue
            
            recommendations.append({
                "concept": target,
                "relation": rel_wrapper.relationship.relation,
                "strength": rel_wrapper.relationship.strength,
                "type": "direct"
            })
        
        # Get incoming relationships
        for source, _, data in concept_graph.graph.in_edges(concept_name, data=True):
            rel_wrapper = data.get('relationship')
            if not rel_wrapper:
                continue
            
            if user_state.confidence.get(source, 0.0) > 0.8:
                continue
            
            recommendations.append({
                "concept": source,
                "relation": f"inverse_{rel_wrapper.relationship.relation}",
                "strength": rel_wrapper.relationship.strength,
                "type": "inverse"
            })
        
        # Score and sort
        for rec in recommendations:
            confidence = user_state.confidence.get(rec["concept"], 0.0)
            rec["score"] = rec["strength"] * (1.0 - confidence * 0.5)
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations[:limit]

    def recommend_for_review(self, limit: int = 5) -> List[Dict]:
        """Recommend concepts that need review"""
        user_state = self.user_manager.get_user_knowledge()
        
        review_candidates = []
        
        for concept_name, confidence in user_state.confidence.items():
            # Concepts with medium confidence need review
            if 0.3 < confidence < 0.7:
                exposure = user_state.exposure.get(concept_name, 0)
                last_seen = user_state.last_seen.get(concept_name, 0)
                
                review_candidates.append({
                    "concept": concept_name,
                    "confidence": confidence,
                    "exposure": exposure,
                    "last_seen": last_seen,
                    "review_priority": (0.5 - abs(confidence - 0.5)) * (1.0 / max(1, exposure))
                })
        
        review_candidates.sort(key=lambda x: x["review_priority"], reverse=True)
        
        return review_candidates[:limit]

    def discover_new_concepts(self, limit: int = 5) -> List[str]:
        """Discover new concepts the user hasn't encountered"""
        concept_graph = self.knowledge_store.graph
        user_state = self.user_manager.get_user_knowledge()
        
        known_concepts = set(user_state.confidence.keys())
        all_concepts = set(concept_graph.graph.nodes())
        
        new_concepts = all_concepts - known_concepts
        
        # Score new concepts by graph centrality
        scored_concepts = []
        for concept in new_concepts:
            if concept in concept_graph.graph:
                degree = concept_graph.graph.degree(concept)
                scored_concepts.append((concept, degree))
        
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return [c[0] for c in scored_concepts[:limit]]