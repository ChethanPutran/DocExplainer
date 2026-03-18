from typing import List, Optional
from src.core.knowledge.models.graph import ConceptGraph
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.delta import GraphDelta
from .chain import BaseDocumentChain

class KnowledgeRepository:
    """Repository for knowledge graph persistence"""
    
    def __init__(self, document_chain: BaseDocumentChain, concept_graph: ConceptGraph):
        self.chain = document_chain
        self.graph = concept_graph

    def save_delta(self, delta: GraphDelta):
        """Save a delta to the chain"""
        self.chain.add(delta.section_id, delta)

    def get_deltas_upto(self, section_id: int) -> List[GraphDelta]:
        """Get deltas up to section id"""
        return self.chain.get_concept_graph_upto(section_id)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        if self.graph.has_concept(name):
            return self.graph.get_concept(name).primary_concept
        return None

    def get_concept_graph(self) -> ConceptGraph:
        """Get the concept graph"""
        return self.graph