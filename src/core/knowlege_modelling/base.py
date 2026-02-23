from time import time
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from src.core.document.document_structures import DocumentChunk


class Concept:
    def __init__(self,
                 name: str,
                 definition: str = "", score: float = 0.0,
                 frequency: int = 0,
                 first_pos: int = -1,
                 embedding = None,
                 attributes: Dict | None = None
                 ):
        self.id = int(time() * 1000)  # Generate a unique integer ID
        self.name = name
        self.score = score
        self.frequency = frequency
        self.first_position = first_pos
        self.aliases = []
        self.embedding = embedding
        self.definitions = [definition] if definition else []
        self.attributes = attributes if attributes is not None else {}
        self.occurrences: List[Dict] = []


class ConceptRelationship:
    def __init__(self,
                 concept1: Concept=None,
                 concept2: Concept=None,
                 definition: str = "",
                 attributes: Dict | None = None,
                 relation: str = "related_to",
                 strength: float = 1.0
                 ):
        self.concept1 = concept1
        self.concept2 = concept2
        self.definition = definition
        self.attributes = attributes if attributes is not None else {}
        self.relation = relation
        self.strength = strength  # document-level


class ConceptNode:
    def __init__(self, primary_concept: Concept,
                 embedding: None = None
                 ):
        self.primary_concept = primary_concept
        self.embedding = embedding  # For future use GNN


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

        # Inside your ConceptGraph class (base.py)
    def add_concept(self, node: ConceptNode):
        # Use the name string as the actual key in the NetworkX graph
        self.graph.add_node(node.primary_concept.name, data=node)

    def update_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        # Extract the canonical string names used as keys in the graph
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name

        if self.graph.has_edge(u_name, v_name):
            # Access the edge data using the string keys
            edge_data = self.graph[u_name][v_name]

            if 'relationship' in edge_data:
                # Increment the strength of the underlying ConceptRelationship
                existing_rel_wrapper = edge_data['relationship']
                existing_rel_wrapper.relationship.strength += relationship.relationship.strength
            else:
                # Handle cases where an edge exists but lacks the custom data object
                self.graph[u_name][v_name]['relationship'] = relationship
        else:
            self.upsert_relationship(concept1, concept2, relationship)

    def upsert_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        try:
            self.update_relationship(concept1, concept2, relationship)
        except ValueError:
            self.add_relationship(concept1, concept2, relationship)

    def add_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name

        if not self.has_concept(u_name):
            self.add_concept(concept1)
        if not self.has_concept(v_name):
            self.add_concept(concept2)

        if self.graph.has_edge(u_name, v_name):
            existing_rel_wrapper = self.graph[u_name][v_name]['relationship']
            existing_rel_wrapper.relationship.strength += relationship.relationship.strength
        else:
            self.graph.add_edge(u_name, v_name, relationship=relationship)

    def remove_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptNodeRelationship):
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name

        if not self.has_concept(u_name):
            self.add_concept(concept1)
        if not self.has_concept(v_name):
            self.add_concept(concept2)

        if self.graph.has_edge(u_name, v_name):
            self.graph.remove_edge(u_name, v_name)

    def get_concept(self, concept_name: str) -> ConceptNode:
        # Now we can do a direct O(1) lookup instead of a loop
        if concept_name in self.graph:
            return self.graph.nodes[concept_name]['data']
        raise ValueError("No concept found")

    def has_concept(self, concept_name: str):
        return concept_name in self.graph

    def print(self):
        if not self.graph.nodes:
            print("The Concept Graph is empty.")
            return

        print("--- Concept Graph Summary ---")
        print(f"Total Concepts: {self.graph.number_of_nodes()}")
        print(f"Total Relationships: {self.graph.number_of_edges()}")
        print("-" * 30)

        # .nodes(data=True) returns (node_id, attribute_dict)
        for node_name, node_attrs in self.graph.nodes(data=True):
            # Retrieve the ConceptNode object from the 'data' key
            node_obj = node_attrs.get('data')
            
            # Now we can safely access the object's properties
            print(f"Concept: {node_name}")

            # Find all outgoing edges from this string key
            edges = self.graph.out_edges(node_name, data=True)
            
            if not edges:
                print("  -> No outgoing relationships.")
            else:
                for _, target_name, data in edges:
                    rel_wrapper = data.get('relationship')
                    
                    # Check if relationship data exists to avoid NoneType errors
                    if rel_wrapper:
                        strength = rel_wrapper.relationship.strength
                        # Using .relation based on your ConceptRelationship class definition
                        rel_type = getattr(rel_wrapper.relationship, 'relation', 'related_to')
                        
                        print(f"  --[{rel_type} (w={strength:.2f})]--> {target_name}")
        
        print("-" * 30)

    def get_dependencies(self, concept: Concept) -> List[Concept]:
        """
        Returns list of Concept objects that the given concept depends on.
        That means:
            concept --depends_on--> dependency
        """

        dependencies = []

        concept_name = concept.name

        if concept_name not in self.graph:
            return dependencies

        # Iterate over outgoing edges
        for _, target_name, edge_data in self.graph.out_edges(concept_name, data=True):

            rel_wrapper = edge_data.get("relationship")

            if not rel_wrapper:
                continue

            relation_type = getattr(rel_wrapper.relationship, "relation", "")

            if relation_type == "depends_on":
                target_node = self.graph.nodes[target_name]["data"]
                dependencies.append(target_node.primary_concept)

        return dependencies

    def visualize(self):
        print("Visualizing Concept Graph...")
        labels = {node: node for node in self.graph.nodes()}
        pos = nx.spring_layout(self.graph)

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos,
                labels=labels,       # Use the mapping dictionary
                with_labels=True,
                node_size=2000,
                node_color="lightblue",
                font_size=10)
        plt.show()

    def find_prerequisites(self, target_concept: str, user_knowledge: Set[str]) -> List[str]:
        """Find prerequisite concepts for a target concept."""
        if target_concept not in self.graph:
            return []

        visited = set()
        queue = [(target_concept, 0)]
        prerequisites = []

        while queue:
            concept, depth = queue.pop(0)
            if concept in visited:
                continue

            visited.add(concept)

            for predecessor in self.graph.predecessors(concept):
                edge_data = self.graph.get_edge_data(predecessor, concept)

                if edge_data and edge_data.get("relation") in [
                    "is_a",
                    "part_of",
                    "prerequisite",
                ]:
                    if predecessor not in user_knowledge:
                        prerequisites.append(
                            {
                                "concept": predecessor,
                                "depth": depth + 1,
                                "relation": edge_data.get("relation"),
                                "importance": self.graph.nodes[predecessor].get("score", 0),
                            }
                        )

                    if depth < 3:
                        queue.append((predecessor, depth + 1))

        prerequisites.sort(key=lambda x: (-x["importance"], x["depth"]))

        return prerequisites[:10]


class GraphDelta:
    def __init__(self, section_id: int, data: DocumentChunk):
        self.section_id = section_id
        # Changed key to str (name)
        self.new_concepts: Dict[str, ConceptNode] = {}
        self.new_edges: List[ConceptNodeRelationship] = []
        self.edge_updates: Dict[Tuple[str, str], ConceptNodeRelationship] = {}
        self.data = data

    def create(self, G: ConceptGraph, concept_list: List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]):
        # Local tracking to avoid duplicating a new concept within the same delta
        local_nodes: Dict[str, ConceptNode] = {}

        for main_concept, related_concepts in concept_list:
            # 1. Resolve Main Node: check Global Graph -> Local Cache -> Create New
            main_node = G.get_concept(
                main_concept.name) or local_nodes.get(main_concept.name)
            if not main_node:
                main_node = ConceptNode(primary_concept=main_concept)
                local_nodes[main_concept.name] = main_node
                self.new_concepts[main_concept.name] = main_node

            for rel_concept, rel_obj in related_concepts:
                # 2. Resolve Related Node
                rel_node = G.get_concept(
                    rel_concept.name) or local_nodes.get(rel_concept.name)
                if not rel_node:
                    rel_node = ConceptNode(primary_concept=rel_concept)
                    local_nodes[rel_concept.name] = rel_node
                    self.new_concepts[rel_concept.name] = rel_node

                # 3. Build Node Relationship
                node_rel = ConceptNodeRelationship(
                    main_node, rel_node, rel_obj)

                u_name, v_name = main_node.primary_concept.name, rel_node.primary_concept.name
                if G.graph.has_edge(u_name, v_name):
                    self.edge_updates[(u_name, v_name)] = node_rel
                else:
                    self.new_edges.append(node_rel)
