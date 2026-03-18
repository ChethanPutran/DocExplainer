from typing import List, Dict, Optional
from ..models.graph import ConceptGraph
from ..models.concept import Concept
from ..models.relationship import ConceptRelationship, ConceptNode
from .concept_factory import ConceptFactory
from .relationship_factory import RelationshipFactory

class GraphFactory:
    """Factory for creating and building concept graphs"""
    
    def __init__(self,
                 concept_factory: Optional[ConceptFactory] = None,
                 relationship_factory: Optional[RelationshipFactory] = None):
        self.concept_factory = concept_factory or ConceptFactory()
        self.relationship_factory = relationship_factory or RelationshipFactory()
    
    def create_empty_graph(self) -> ConceptGraph:
        """Create an empty concept graph"""
        return ConceptGraph()
    
    def build_graph_from_concepts(self,
                                 concepts: List[Concept],
                                 relationships: List[ConceptRelationship]) -> ConceptGraph:
        """Build a graph from concepts and relationships"""
        graph = ConceptGraph()
        
        # Add concept nodes
        for concept in concepts:
            node = self.relationship_factory.create_node(concept)
            graph.add_concept_node(node)
        
        # Add relationships
        for rel in relationships:
            node1 = graph.get_concept(rel.concept1.name)
            node2 = graph.get_concept(rel.concept2.name)
            
            if node1 and node2:
                node_rel = self.relationship_factory.create_node_relationship(node1, node2, rel)
                graph.add_relationship(node1, node2, node_rel)
        
        return graph
    
    def merge_graphs(self, graphs: List[ConceptGraph]) -> ConceptGraph:
        """Merge multiple graphs into one"""
        merged = ConceptGraph()
        
        for graph in graphs:
            for node_name, node_data in graph.graph.nodes(data=True):
                if not merged.has_concept(node_name):
                    merged.add_concept_node(node_data['data'])
            
            for u, v, data in graph.graph.edges(data=True):
                if merged.graph.has_edge(u, v):
                    continue
                
                node_u = merged.get_concept(u)
                node_v = merged.get_concept(v)
                if node_u and node_v and 'relationship' in data:
                    merged.add_relationship(node_u, node_v, data['relationship'])
        
        return merged