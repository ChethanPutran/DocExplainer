from __future__ import annotations
from typing import Dict,TYPE_CHECKING
from src.core.user import UserManager, UserKnowledgeState
from ..repository import BaseKnowledgeStore, BaseKnowledgeRepository
from ..models import ConceptGraph, GraphDelta
from .builder import ConceptGraphBuilder
from .chain import DocumentChain
from .updater import GraphUpdater

if TYPE_CHECKING:
    from src.core.document import DocumentTree

class GraphStateManager:
    """Manages the state of the knowledge graph"""
    
    def __init__(
        self, 
        user_manager: UserManager,
        concept_graph_builder: ConceptGraphBuilder,
        document_chain: DocumentChain,
        graph_updater: GraphUpdater,
        repository: BaseKnowledgeRepository,
        knowledge_store: BaseKnowledgeStore 
    ):
        self.user_manager = user_manager
        self.concept_graph_builder = concept_graph_builder
        self.document_chain = document_chain
        self.repository = repository
        self.graph_updater = graph_updater
        self.knowledge_store = knowledge_store
        self.document = None

    def _sync_user_state(self) -> UserKnowledgeState:
        """Convert user knowledge to graph confidence scores"""
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