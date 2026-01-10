from time import time
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from src.core.document.document_processing import DocumentChunk

class Concept:
    def __init__(self, 
                 name: str,
                 description: str = "", score: float = 0.0,
                 frequency: int = 0,
                 first_pos: int = -1,
                 attributes: Dict | None = None
                    ):
        self.id = int(time() * 1000)  # Generate a unique integer ID
        self.name = name
        self.score = score
        self.frequency = frequency
        self.first_position = first_pos

        self.description = description
        self.attributes = attributes if attributes is not None else {}


class ConceptRelationship:
    def __init__(self, 
                 concept1: Concept,
                 concept2: Concept,
                 description: str = "",
                 attributes: Dict | None = None,
                 relation: str = "related_to",
                 strength: float = 1.0
                 ):
        self.concept1 = concept1
        self.concept2 = concept2
        self.description = description
        self.attributes = attributes if attributes is not None else {}
        self.relation = relation
        self.strength = strength  # document-level


class ConceptNode:
    def __init__(self, primary_concept: Concept,
                  embedding: None = None
                  ):
        self.primary_concept = primary_concept
        self.embedding = embedding # For future use GNN


class ConceptNodeRelationship:
    def __init__(self, 
                 concept1: ConceptNode,
                 concept2: ConceptNode,
                 relationship: ConceptRelationship
                 ):
        self.concept1 = concept1
        self.concept2 = concept2
        self.relationship = relationship


class ConceptGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}

    def add_concept(self, concept: ConceptNode):
        self.graph.add_node(concept)

    def update_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        if self.graph.has_edge(concept1, concept2):
            existing_rel = self.graph[concept1][concept2]['relationship']
            existing_rel.relationship.strength += relationship.relationship.strength
        else:
            raise ValueError("Relationship does not exist between the given concepts.")
        
    def add_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        if not self.graph.has_node(concept1):
            self.add_concept(concept1)
        if not self.graph.has_node(concept2):
            self.add_concept(concept2)
        if self.graph.has_edge(concept1, concept2):
            # Update existing relationship weight
            existing_rel = self.graph[concept1][concept2]['relationship']
            existing_rel.relation += relationship.relationship.strength
        else:
            self.graph.add_edge(concept1, concept2, relationship=relationship)

    def get_concept(self, concept_name: str) -> ConceptNode | None:
        for concept in self.graph.nodes:
            if concept.primary_concept.name == concept_name:
                return concept
        return None
    
    def visualize(self):
        print("Visualizing Concept Graph...")
        print(self.graph.nodes())
        labels = {node: node.primary_concept.name for node in self.graph.nodes()}
        pos = nx.spring_layout(self.graph)
    
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, 
                labels=labels,       # Use the mapping dictionary
                with_labels=True, 
                node_size=2000, 
                node_color="lightblue", 
                font_size=10)
        plt.show()



class GraphDelta:
    """
    Stores ONLY what changes at a document position
    """
    def __init__(self, section_id: int, data: DocumentChunk):
        self.section_id : int = section_id
        self.new_concepts: Dict[int, ConceptNode] = {}        # cid -> ConceptNode
        self.new_edges: List[ConceptNodeRelationship] = []           # List[ConceptEdge]
        self.edge_updates: Dict[Tuple[ConceptNode,  ConceptNode], ConceptNodeRelationship] = {}        # (u, v) -> Δweight
        self.data: DocumentChunk = data        # (u, v) -> Δweight

    def create(self, G: ConceptGraph, concept_list:  List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]) -> None:
        """
        Create graph delta from concept list
        Args:
            G (ConceptGraph): Existing concept graph
            concept_list (List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]): List of main concepts and their related concepts with relationships
        
        """
        for main_concept, related_concepts in concept_list:
            main_node = G.get_concept(main_concept.name)

            # Add main concept if not exists
            if main_node is None:
                main_node = ConceptNode(primary_concept=main_concept)
                self.new_concepts[main_concept.id] = main_node

            # Add related concepts and relationships
            for related_concept, relationship in related_concepts:
                related_node = G.get_concept(related_concept.name)

                # Add related concept if not exists
                if related_node is None:
                    related_node = ConceptNode(primary_concept=related_concept)
                    self.new_concepts[related_concept.id] = related_node
                
                # Create relationship
                concept_relationship = ConceptNodeRelationship(
                    concept1=main_node,
                    concept2=related_node,
                    relationship=relationship
                )
                if G.graph.has_edge(main_node, related_node):
                    self.edge_updates[(main_node, related_node)] = concept_relationship
                else:
                    self.new_edges.append(concept_relationship)
        
        