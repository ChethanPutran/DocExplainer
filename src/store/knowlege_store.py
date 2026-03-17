# src/store/knowlege_store.py

from typing import List, Optional

from src.core.knowlege.base import ConceptGraph, ConceptInvertedIndex, ConceptNode, ConceptNodeRelationship
from abc import ABC, abstractmethod
from src.core.knowlege.base import Concept, ConceptRelationship

class BaseKnowledgeStore(ABC):
    graph = ConceptGraph()  # Shared graph instance for all implementations
    @abstractmethod
    def save_concept(self, node: ConceptNode):
        pass

    @abstractmethod
    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        pass

    @abstractmethod
    def save_relationship(self, edge: ConceptNodeRelationship):
        pass

    @abstractmethod
    def upsert_concepts(self, concepts: List[ConceptNode]):
        pass
    
    @abstractmethod
    def get_inverted_index(self) -> ConceptInvertedIndex:
        pass
        
class KnowledgeStore(BaseKnowledgeStore):
    def __init__(self, storage_path="db/knowledge_graph.gpickle"):
        self.path = storage_path
        self.concepts = {}
        self.relationships = {}
        self.graph = ConceptGraph()
        self.inverted_index = ConceptInvertedIndex()

    def get_inverted_index(self) -> ConceptInvertedIndex:
        return self.inverted_index
    
    def save_concept(self, node: ConceptNode):
        # Store serialized concept data in NetworkX nodes
        self.graph.add_concept_node(node)
        
    def has_concept(self, name):
        return name in self.concepts
    
    def get_concept_graph(self):
        return self.graph
    
    def add_concept(self, concept):
        self.concepts[concept.name] = concept
    
    def save_relationship(self, edge: ConceptNodeRelationship):
        key = (edge.concept1, edge.concept2)
        self.relationships[key] = edge
        self.graph.add_relationship(edge.concept1, edge.concept2, edge)

    def get_concept_by_name(self, name):
        return self.concepts.get(name)
    
    def get_relationship(self, concept1_name, concept2_name):
        key = (concept1_name, concept2_name)
        return self.relationships.get(key)
    
    def get_related_concepts(self, concept, relationship_type=None):
        related_concepts = []
        for (concept1, concept2), relationship in self.relationships.items():
            if concept1 == concept.name:
                related_concepts.append(concept2)
            elif concept2 == concept.name:
                related_concepts.append(concept1)
        return related_concepts

    def upsert_concepts(self, concepts):
        for concept in concepts:
            self.add_concept(concept)

