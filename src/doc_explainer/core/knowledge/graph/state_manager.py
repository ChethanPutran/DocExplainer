from __future__ import annotations
from typing import Dict, TYPE_CHECKING, Any

from doc_explainer.core.user.services.user_manager import UserManager
from doc_explainer.store.document.repository import DocumentRepository
from ..repository import BaseKnowledgeStore, BaseKnowledgeRepository
from ..models import ConceptGraph, GraphDelta
from .builder import ConceptGraphBuilder
from .chain import DocumentChain
from .updater import GraphUpdater

if TYPE_CHECKING:
    from ...document import DocumentTree

class GraphStateManager:
    """Manages the state of the knowledge graph"""
    
    def __init__(
        self, 
        user_manager: UserManager,
        concept_graph_builder: ConceptGraphBuilder,
        document_chain: DocumentChain,
        graph_updater: GraphUpdater,
        repository: BaseKnowledgeRepository,
        knowledge_store: BaseKnowledgeStore,
        document_repository: DocumentRepository, 
    ):
        self.user_manager = user_manager
        self.document_repository = document_repository
        self.concept_graph_builder = concept_graph_builder
        self.document_chain = document_chain
        self.repository = repository
        self.graph_updater = graph_updater
        self.knowledge_store = knowledge_store
        self.document = None
        self.full_graph = None

    def _sync_user_state(self) -> Any:
        """Convert user knowledge to graph confidence scores"""
        from ....core.user import UserKnowledgeState
        
        profile = self.user_manager.get_user().knowledge_state
        state = UserKnowledgeState()
        for concept, s in profile.knowledge_states.items():
            state.confidence[concept.name] = s.p_knowledge
        return state

    def build_chain(self, document: DocumentTree, concepts_per_para: int = 10):
        """Build knowledge chain from document"""
        self.document = document
        
        for section in document.root.children.values():
            # Extract concepts
            self.concept_graph_builder.add_concepts_to_document(
                document, section.id, concepts_per_para
            )
            
            # Persist concepts
            self.knowledge_store.upsert_concepts(section.concepts)
            
            # Create delta
            delta = GraphDelta(section_id=section.id, data=section.chunk)
            delta.create(
                self.knowledge_store.graph, 
                section.concepts, 
                section.concept_relationships
            )

            # Add to chain
            self.document_chain.add(section.id, delta)
            self.repository.save_delta(delta)

    def get_document_context(self, check_point: str) -> Dict:
        """Get document context up to checkpoint"""
        return self.document_chain.get_document_context(check_point)

    def get_concept_graph_upto(self, section_id: str) -> ConceptGraph:
        """Get concept graph up to section"""
        # Sync user state
        self.graph_updater.user_state = self._sync_user_state()
        
        # Build view graph
        view_graph = ConceptGraph()
        
        for delta in self.repository.get_deltas_upto(section_id):
            self.graph_updater.apply_delta(delta, target_graph=view_graph)
            
        return view_graph

    def get_concept_graph(self) -> ConceptGraph:
        """Get full concept graph"""

        if(self.full_graph is not None):
            return self.full_graph
        
        # Sync user state
        self.graph_updater.user_state = self._sync_user_state()
        
        # Build full graph
        full_graph = ConceptGraph()
        
        for delta in self.repository.get_all_deltas():
            self.graph_updater.apply_delta(delta, target_graph=full_graph)

        self.full_graph = full_graph
        return full_graph

    def build_graph_from_document(self, document_id: str) -> ConceptGraph:
        """Build full graph from document"""
        document = self.document_repository.get_document_tree(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        self.build_chain(document)
        return self.get_concept_graph()