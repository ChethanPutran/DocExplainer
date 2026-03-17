import pickle
import json
import networkx as nx
from typing import Optional, List
import os
from src.core.knowledge.models.graph import ConceptGraph
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship
from .concept_repository import ConceptRepository
from .relationship_repository import RelationshipRepository
from .serializers import GraphSerializer


class GraphRepository:
    """Repository for graph persistence"""
    
    def __init__(self, storage_path: str = "data/knowledge/graphs/",
                 concept_repo: Optional[ConceptRepository] = None,
                 relationship_repo: Optional[RelationshipRepository] = None):
        self.storage_path = storage_path
        self.concept_repo = concept_repo or ConceptRepository()
        self.relationship_repo = relationship_repo or RelationshipRepository()
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_graph(self, graph: ConceptGraph, name: str = "default") -> str:
        """Save graph to file"""
        # Save in multiple formats
        
        # 1. Pickle format (full object)
        pickle_path = os.path.join(self.storage_path, f"{name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(graph, f)
        
        # 2. NetworkX format (graph only)
        nx_path = os.path.join(self.storage_path, f"{name}.gpickle")
        nx.write_gpickle(graph.graph, nx_path)
        
        # 3. JSON format (serialized)
        json_path = os.path.join(self.storage_path, f"{name}.json")
        with open(json_path, 'w') as f:
            json.dump(GraphSerializer.serialize_graph(graph), f, indent=2)
        
        return pickle_path
    
    def load_graph(self, name: str = "default") -> Optional[ConceptGraph]:
        """Load graph from file"""
        pickle_path = os.path.join(self.storage_path, f"{name}.pkl")
        
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        
        # Try loading from JSON
        json_path = os.path.join(self.storage_path, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return GraphSerializer.deserialize_graph(data)
        
        return None
    
    def export_graphml(self, graph: ConceptGraph, name: str = "default") -> str:
        """Export graph to GraphML format"""
        path = os.path.join(self.storage_path, f"{name}.graphml")
        nx.write_graphml(graph.graph, path)
        return path
    
    def import_graphml(self, filepath: str) -> ConceptGraph:
        """Import graph from GraphML"""
        graph_nx = nx.read_graphml(filepath)
        graph = ConceptGraph()
        graph.graph = graph_nx
        return graph
    
    def delete_graph(self, name: str = "default") -> bool:
        """Delete saved graph"""
        formats = ['pkl', 'gpickle', 'json', 'graphml']
        deleted = False
        
        for ext in formats:
            path = os.path.join(self.storage_path, f"{name}.{ext}")
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        
        return deleted
    
    def list_graphs(self) -> List[str]:
        """List available graphs"""
        graphs = set()
        
        for filename in os.listdir(self.storage_path):
            name, ext = os.path.splitext(filename)
            if ext in ['.pkl', '.gpickle', '.json', '.graphml']:
                graphs.add(name)
        
        return list(graphs)
    
    def build_graph_from_repositories(self) -> ConceptGraph:
        """Build a graph from concept and relationship repositories"""
        graph = ConceptGraph()
        
        # Add all concepts
        concepts = self.concept_repo.get_all_concepts()
        for concept in concepts:
            if not graph.has_concept(concept.name):
                # Create node
                from src.core.knowledge.models.relationship import ConceptNode
                node = ConceptNode(primary_concept=concept)
                graph.add_concept_node(node)
        
        # Add all relationships
        relationships = self.relationship_repo.get_all_relationships()
        for rel in relationships:
            node1 = graph.get_concept(rel.concept1.name)
            node2 = graph.get_concept(rel.concept2.name)
            
            if node1 and node2:
                from src.core.knowledge.models.relationship import ConceptNodeRelationship
                node_rel = ConceptNodeRelationship(node1, node2, rel)
                graph.add_relationship(node1, node2, node_rel)
        
        return graph