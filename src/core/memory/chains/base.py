from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from ..base.interfaces import ChainInterface


class BaseChain(ABC, ChainInterface):
    """Base class for chain structures"""
    
    def __init__(self):
        self.nodes: Dict[Any, Any] = {}
        self.edges: Dict[Any, List[Any]] = {}
    
    def add_node(self, node_id: Any, data: Any) -> bool:
        """Add a node to the chain"""
        if node_id not in self.nodes:
            self.nodes[node_id] = data
            self.edges[node_id] = []
            return True
        return False
    
    def get_node(self, node_id: Any) -> Optional[Any]:
        """Get a node from the chain"""
        return self.nodes.get(node_id)
    
    def add_edge(self, from_node: Any, to_node: Any):
        """Add an edge between nodes"""
        if from_node in self.edges and to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)
    
    def get_edges(self, node_id: Any) -> List[Any]:
        """Get edges from a node"""
        return self.edges.get(node_id, [])
    
    def traverse(self, start_node: Any) -> list:
        """Traverse the chain from start node"""
        result = []
        visited = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            if node in self.nodes:
                result.append(self.nodes[node])
            
            # Add children in reverse order for proper traversal
            for child in reversed(self.edges.get(node, [])):
                if child not in visited:
                    stack.append(child)
        
        return result
    
    def clear(self):
        """Clear the chain"""
        self.nodes.clear()
        self.edges.clear()