from typing import Any, Dict, List, Optional, Tuple
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

    def get_dependents(self, concept_name: str) -> List[Concept]:
        """Get dependents of a concept"""
        if concept_name not in self.graph:
            return []
        return [self.graph.nodes[successor]['data'].primary_concept for successor in self.graph.successors(concept_name)]

    def get_prerequisites(self, concept_name: str, user_confidence: float) -> List[Dict[str, Any]]:
        """Get prerequisites for a concept based on user confidence"""
        if concept_name not in self.graph:
            return []
        
        prerequisites = []
        for predecessor in self.graph.predecessors(concept_name):
            edge_data = self.graph[predecessor][concept_name]['data']
            if edge_data.relationship.strength > user_confidence:
                prerequisites.append({
                    "concept": predecessor,
                    "relationship_strength": edge_data.relationship.strength
                })
        return prerequisites


    def get_dependencies(self, concept: ConceptNode) -> List[ConceptNode]:
        """Get dependencies of a concept node"""
        if concept.primary_concept.name not in self.graph:
            return []
        return [self.graph.nodes[predecessor]['data'] for predecessor in self.graph.predecessors(concept.primary_concept.name)]