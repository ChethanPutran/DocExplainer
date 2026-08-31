from typing import List

from doc_explainer.core.knowledge.models import Concept, ConceptRelationship

from ...core.knowledge.models.relationship import ConceptNode, ConceptNodeRelationship
from ...core.knowledge.models import ConceptGraph, ConceptInvertedIndex
from ...core.knowledge import BaseKnowledgeStore
        
class KnowledgeStore(BaseKnowledgeStore):
    def __init__(self, storage_path="db/knowledge_graph.gpickle"):
        self.path = storage_path
        self.concepts = {}
        self.relationships = {}
        self.graph = ConceptGraph()
        self.inverted_index = ConceptInvertedIndex()

    def get_inverted_index(self) -> ConceptInvertedIndex:
        return self.inverted_index
    
    def save_concept(self, node: Concept):
        # Store serialized concept data in NetworkX nodes
        raise NotImplementedError("This method should be implemented in a subclass.")

    def save_concept_node(self, node: ConceptNode):
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

    def get_concept_node_by_name(self, name: str) -> ConceptNode | None:
        """Get concept node by name"""
        return self.graph.get_concept(name)

    def save_concept_relationship(self, edge: ConceptRelationship):
        """Save concept relationship"""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def save_concept_node_relationship(self, edges: ConceptNodeRelationship):
        """Save concept node relationship"""
        self.graph.add_relationship(edges.concept1, edges.concept2, edges)

    def get_dependents(self, concept_name: str) -> List[Concept]:
        """Get dependents of a concept"""
        dependents = []
        for (concept1, concept2), relationship in self.relationships.items():
            if concept1 == concept_name:
                dependents.append(self.get_concept_by_name(concept2))
            elif concept2 == concept_name:
                dependents.append(self.get_concept_by_name(concept1))
        return dependents


