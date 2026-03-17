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

        if self.graph.has_edge(u_name, v_name):
            existing_rel = self.graph[u_name][v_name]['relationship']
            existing_rel.relationship.strength += relationship.relationship.strength
        else:
            self.graph.add_edge(u_name, v_name, relationship=relationship)

    def update_relationship(self, concept1: ConceptNode, concept2: ConceptNode,
                          relationship: ConceptNodeRelationship):
        """Update an existing relationship"""
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name

        if self.graph.has_edge(u_name, v_name):
            edge_data = self.graph[u_name][v_name]
            if 'relationship' in edge_data:
                existing_rel = edge_data['relationship']
                existing_rel.relationship.strength += relationship.relationship.strength
            else:
                self.graph[u_name][v_name]['relationship'] = relationship
        else:
            self.add_relationship(concept1, concept2, relationship)

    def remove_relationship(self, concept1: ConceptNode, concept2: ConceptNode):
        """Remove a relationship between concepts"""
        u_name = concept1.primary_concept.name
        v_name = concept2.primary_concept.name
        
        if self.graph.has_edge(u_name, v_name):
            self.graph.remove_edge(u_name, v_name)

    def get_dependencies(self, concept: Concept) -> List[Concept]:
        """Get concepts that this concept depends on"""
        dependencies = []
        concept_name = concept.name

        if concept_name not in self.graph:
            return dependencies

        for _, target_name, edge_data in self.graph.out_edges(concept_name, data=True):
            rel_wrapper = edge_data.get("relationship")
            if not rel_wrapper:
                continue

            relation_type = getattr(rel_wrapper.relationship, "relation", "")
            if relation_type == "depends_on":
                target_node = self.graph.nodes[target_name]["data"]
                dependencies.append(target_node.primary_concept)

        return dependencies

    def get_dependents(self, concept: Concept) -> List[Concept]:
        """Get concepts that depend on this concept"""
        dependents = []
        concept_name = concept.name

        if concept_name not in self.graph:
            return dependents

        for source_name, _, edge_data in self.graph.in_edges(concept_name, data=True):
            rel_wrapper = edge_data.get("relationship")
            if not rel_wrapper:
                continue

            relation_type = getattr(rel_wrapper.relationship, "relation", "")
            if relation_type == "depends_on":
                source_node = self.graph.nodes[source_name]["data"]
                dependents.append(source_node.primary_concept)

        return dependents

    def get_prerequisites(self, target_concept: str, user_knowledge: Dict) -> List[Dict]:
        """Find prerequisite concepts for a target concept"""
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
                rel_wrapper = edge_data.get('relationship') if edge_data else None
                
                if rel_wrapper and rel_wrapper.relationship.relation in ["is_a", "part_of", "depends_on"]:
                    if predecessor not in user_knowledge:
                        node_data = self.graph.nodes[predecessor].get('data')
                        score = node_data.primary_concept.score if node_data else 0
                        
                        prerequisites.append({
                            "concept": predecessor,
                            "depth": depth + 1,
                            "relation": rel_wrapper.relationship.relation,
                            "importance": score,
                        })

                    if depth < 3:
                        queue.append((predecessor, depth + 1))

        prerequisites.sort(key=lambda x: (-x["importance"], x["depth"]))
        return prerequisites[:10]

    def print_summary(self):
        """Print graph summary"""
        if not self.graph.nodes:
            print("The Concept Graph is empty.")
            return

        print("--- Concept Graph Summary ---")
        print(f"Total Concepts: {self.graph.number_of_nodes()}")
        print(f"Total Relationships: {self.graph.number_of_edges()}")
        print("-" * 30)

        for node_name, node_attrs in self.graph.nodes(data=True):
            node_obj = node_attrs.get('data')
            if node_obj:
                print(f"Concept: {node_name}")
                print(f"  Score: {node_obj.primary_concept.score:.2f}")
                print(f"  Frequency: {node_obj.primary_concept.frequency}")

                edges = self.graph.out_edges(node_name, data=True)
                if edges:
                    for _, target_name, data in edges:
                        rel_wrapper = data.get('relationship')
                        if rel_wrapper:
                            strength = rel_wrapper.relationship.strength
                            rel_type = rel_wrapper.relationship.relation
                            print(f"  --[{rel_type} (w={strength:.2f})]--> {target_name}")
        print("-" * 30)