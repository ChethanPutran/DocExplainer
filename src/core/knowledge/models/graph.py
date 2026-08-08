from typing import Dict, List, Optional, Tuple
import networkx as nx
from .concept import Concept
from .relationship import ConceptNode, ConceptNodeRelationship

class ConceptGraph:
    """Knowledge graph of concepts and their relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}

    def add_concept_node(self, node: ConceptNode):
        """Add a concept node to the graph"""
        self.graph.add_node(node.primary_concept.name, data=node)

    def has_concept(self, concept_name: str) -> bool:
        """Check if concept exists in graph"""
        return concept_name in self.graph

    def update_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        """Update relationship between two concepts"""
        if self.graph.has_edge(concept1.primary_concept.name, concept2.primary_concept.name):
            self.graph[concept1.primary_concept.name][concept2.primary_concept.name]['data'] = relationship
            
    def get_concept(self, concept_name: str) -> Optional[ConceptNode]:
        """Get concept node by name"""
        if concept_name in self.graph:
            return self.graph.nodes[concept_name]['data']
        return None

    def add_relationship(self, concept1: ConceptNode, concept2: ConceptNode, 
                        relationship: ConceptNodeRelationship):
        """Add a relationship between concepts"""
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name

        if not self.has_concept(u_name):
            self.add_concept_node(concept1)
        if not self.has_concept(v_name):
            self.add_concept_node(concept2)
        
        self.graph.add_edge(u_name, v_name, data=relationship)