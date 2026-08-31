from __future__ import annotations
import pickle
import json
import networkx as nx
from typing import Optional, List, TYPE_CHECKING
import os

from ...core.knowledge import ConceptInvertedIndex, ConceptNodeRelationship

from .concept_repository import ConceptRepository



from .serializers import GraphSerializer
from .concept_repository import ConceptRepository
from .relationship_repository import RelationshipRepository

from ...core.knowledge import (
    BaseDocumentChain,
    GraphDelta ,
    ConceptGraph,
      ConceptNode,
      BaseKnowledgeRepository)

class KnowledgeRepository(BaseKnowledgeRepository):
    """Repository for graph persistence"""
    def __init__(self,
                 document_chain: BaseDocumentChain, 
                 concept_graph: ConceptGraph,
                 storage_path: str = "data/knowledge/graphs/",
                 concept_repo: Optional['ConceptRepository'] = None,
                 relationship_repo: Optional['RelationshipRepository'] = None,
                 ):
        self.storage_path = storage_path

        if concept_repo is None:
            concept_repo = ConceptRepository()
        if relationship_repo is None:
            relationship_repo = RelationshipRepository()

        self.concept_repo = concept_repo
        self.relationship_repo = relationship_repo
        self.full_graph = None
        self.chain = document_chain
        self.graph = concept_graph
        self._ensure_storage()
    
    def save_delta(self, delta: GraphDelta):
        """Save a delta to the chain"""
        self.chain.add(delta.section_id, delta)

    def get_deltas_upto(self, section_id: str) -> List[GraphDelta]:
        """Get deltas up to section id"""
        return self.chain.get_concept_graph_upto(section_id)

    def get_all_deltas(self) -> List[GraphDelta]:
        """Get all deltas"""
        return self.chain.get_all_deltas()

    def get_concept_node_by_name(self, name: str) -> Optional[ConceptNode]:
        """Get concept node by name"""
        if self.graph.has_concept(name):
            return self.graph.get_concept(name)
        return None
    
    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        pass

    def get_concept_graph(self) -> ConceptGraph:
        """Get the concept graph"""
        return self.graph
    
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
    
    def get_inverted_index(self) -> ConceptInvertedIndex:
        return super().get_inverted_index()
    
    def update_graph(self, graph: ConceptGraph):
        """Update the graph with new data"""
        self.graph = graph
        self.save_graph(graph)

    def upsert_concepts(self, concepts: List[ConceptNode]):
        return super().upsert_concepts(concepts)
    
    def build_graph_from_repositories(self) -> ConceptGraph:
        """Build a graph from concept and relationship repositories"""
        graph = ConceptGraph()
        
        # Add all concepts
        concepts = self.concept_repo.get_all_concepts()
        for concept in concepts:
            if not graph.has_concept(concept.name):
                # Create node
                from ...core.knowledge.models.relationship import ConceptNode
                node = ConceptNode(primary_concept=concept)
                graph.add_concept_node(node)
        
        # Add all relationships
        relationships = self.relationship_repo.get_all_relationships()
        for rel in relationships:
            node1 = graph.get_concept(rel.concept1.name)
            node2 = graph.get_concept(rel.concept2.name)
            
            if node1 and node2:
                from ...core.knowledge.models.relationship import ConceptNodeRelationship
                node_rel = ConceptNodeRelationship(node1, node2, rel)
                graph.add_relationship(node1, node2, node_rel)
        
        return graph
