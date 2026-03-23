from typing import List, Optional
from abc import ABC, abstractmethod

from ..models import ConceptGraph, ConceptInvertedIndex, Concept, ConceptRelationship

class BaseKnowledgeStore(ABC):
    graph = ConceptGraph()  # Shared graph instance for all implementations
    @abstractmethod
    def save_concept(self, node: Concept):
        pass

    @abstractmethod
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        pass

    @abstractmethod
    def save_relationship(self, edge: ConceptRelationship):
        pass

    @abstractmethod
    def upsert_concepts(self, concepts: List[Concept]):
        pass
    
    @abstractmethod
    def get_inverted_index(self) -> ConceptInvertedIndex:
        pass