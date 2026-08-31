from __future__ import annotations
from typing import Optional, Any
from ..models.context import Context, SessionContext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...knowledge.models import ConceptGraph
    from ...user import UserKnowledgeState


class ContextManager:
    """Manages comprehensive context for operations"""
    
    def __init__(self, user_knowledge: Optional['UserKnowledgeState'] = None,
                 session_context: Optional[SessionContext] = None,
                 document_context: Optional[Any] = None,
                 concept_graph: Optional['ConceptGraph'] = None):
        if user_knowledge is None:
            from ...user import UserKnowledgeState
            user_knowledge = UserKnowledgeState()
        if session_context is None:            
            session_context = SessionContext()
        if concept_graph is None:
            from src.core.knowledge import ConceptGraph
            concept_graph = ConceptGraph()

        self.user_knowledge = user_knowledge 
        self.session_context = session_context 
        self.document_context = document_context
        self.concept_graph = concept_graph 
    
    def build_context(self) -> Context:
        """Build comprehensive context"""
        return Context(
            user_knowledge=self.user_knowledge,
            session_context=self.session_context,
            document_context=self.document_context,
            concept_graph=self.concept_graph
        )
    
    def update_user_knowledge(self, knowledge_state: UserKnowledgeState):
        """Update user knowledge"""
        self.user_knowledge = knowledge_state
    
    def update_session_context(self, session_context: SessionContext):
        """Update session context"""
        self.session_context = session_context
    
    def update_document_context(self, document_context: Any):
        """Update document context"""
        self.document_context = document_context
    
    def update_concept_graph(self, concept_graph: ConceptGraph):
        """Update concept graph"""
        self.concept_graph = concept_graph
    
    def get_context_for_explanation(self, concept: str) -> Dict:
        """Get context specific for explaining a concept"""
        return {
            "user_knowledge": self.user_knowledge.get_confidence(concept),
            "recent_interactions": self.session_context.get_recent_interactions(3),
            "related_concepts": self._get_related_concepts(concept),
            "has_document_context": self.document_context is not None
        }
    
    def _get_related_concepts(self, concept: str) -> list:
        """Get concepts related to the given concept"""
        if not self.concept_graph or not self.concept_graph.has_concept(concept):
            return []
        
        related = []
        # Get outgoing edges
        for _, target, data in self.concept_graph.graph.out_edges(concept, data=True):
            rel_wrapper = data.get('relationship')
            if rel_wrapper:
                related.append({
                    "concept": target,
                    "relation": rel_wrapper.relationship.relation,
                    "strength": rel_wrapper.relationship.strength
                })
        
        return related[:5]  # Return top 5