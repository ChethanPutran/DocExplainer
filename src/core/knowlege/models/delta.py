from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from src.core.document import DocumentChunk
from .concept import Concept
from .relationship import ConceptNode, ConceptNodeRelationship, ConceptRelationship
from .graph import ConceptGraph

@dataclass
class GraphDelta:
    """Represents changes to the graph from processing a section"""
    section_id: int
    data: DocumentChunk
    new_concepts: Dict[str, ConceptNode] = field(default_factory=dict)
    new_edges: List[ConceptNodeRelationship] = field(default_factory=list)
    edge_updates: Dict[Tuple[str, str], ConceptNodeRelationship] = field(default_factory=dict)

    def create(self, G: ConceptGraph, concept_list: List[Concept], 
               concept_relations: List['ConceptRelationship']):
        """Create delta from concepts and relationships"""
        local_nodes: Dict[str, ConceptNode] = {}

        # Add new concepts
        for concept in concept_list:
            if concept.name not in local_nodes and not G.has_concept(concept.name):
                node = ConceptNode(primary_concept=concept)
                local_nodes[concept.name] = node
                self.new_concepts[concept.name] = node

        # Add relationships
        for relation in concept_relations:
            main_concept = relation.concept1

            # Resolve main node
            main_node = G.get_concept(main_concept.name) or local_nodes.get(main_concept.name)
            if not main_node:
                main_node = ConceptNode(primary_concept=main_concept)
                local_nodes[main_concept.name] = main_node
                self.new_concepts[main_concept.name] = main_node

            # Handle related concept
            rel_concept = relation.concept2
            rel_node = G.get_concept(rel_concept.name) or local_nodes.get(rel_concept.name)
            if not rel_node:
                rel_node = ConceptNode(primary_concept=rel_concept)
                local_nodes[rel_concept.name] = rel_node
                self.new_concepts[rel_concept.name] = rel_node

            # Create relationship
            node_rel = ConceptNodeRelationship(main_node, rel_node, relation)

            u_name, v_name = main_node.primary_concept.name, rel_node.primary_concept.name
            if G.graph.has_edge(u_name, v_name):
                self.edge_updates[(u_name, v_name)] = node_rel
            else:
                self.new_edges.append(node_rel)