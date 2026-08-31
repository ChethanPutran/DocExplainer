from typing import List, Optional
from abc import ABC, abstractmethod

from ...knowledge.models.index import ConceptInvertedIndex
from ...knowledge.models.graph import ConceptGraph,ConceptNode
from ...knowledge.models.delta import GraphDelta


class BaseKnowledgeRepository(ABC):
    """Repository for knowledge graph persistence"""
    graph = ConceptGraph()  # Shared graph instance for all implementations

    @abstractmethod
    def save_delta(self, delta: GraphDelta):
        """Save a delta to the chain"""
        pass

    @abstractmethod
    def get_deltas_upto(self, section_id: str) -> List[GraphDelta]:
        """Get deltas up to section id"""
        pass

    @abstractmethod
    def get_concept_graph(self) -> ConceptGraph:
        """Get the concept graph"""
        pass

    @abstractmethod
    def get_concept_node_by_name(self, name: str) -> Optional[ConceptNode]:
        pass

    @abstractmethod
    def upsert_concepts(self, concepts: List[ConceptNode]):
        pass

    @abstractmethod
    def get_inverted_index(self) -> ConceptInvertedIndex:
        pass

    @abstractmethod
    def get_all_deltas(self) -> List[GraphDelta]:
        """Get all deltas"""
        pass
