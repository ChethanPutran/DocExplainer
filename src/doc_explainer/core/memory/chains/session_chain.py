from typing import Dict, Optional, List
from .base import BaseChain


class SessionChain(BaseChain):
    """Represents a session graph for tracking user interactions"""
    
    def __init__(self):
        super().__init__()
        self.current_node = 0
        self.branch_heads = [0]
        
        # Initialize with root node
        self.add_node(0, {"type": "root", "data": None})
    
    def add_interaction(self, name: str, interaction: Dict, branch: Optional[int] = None):
        """Add an interaction to the session graph"""
        # Create new node
        new_node_id = len(self.nodes)
        
        # Determine parent
        if branch is not None:
            parent = branch
            self.current_node = branch
        else:
            parent = self.current_node
        
        # Add node with interaction data
        node_data = {
            "type": "interaction",
            "name": name,
            "data": interaction,
            "timestamp": interaction.get("timestamp")
        }
        self.add_node(new_node_id, node_data)
        
        # Add edge from parent to new node
        self.add_edge(parent, new_node_id)
        
        # Update current node
        self.current_node = new_node_id
    
    def get_graph(self) -> Dict:
        """Retrieve the session graph"""
        return {
            "nodes": {str(k): v for k, v in self.nodes.items()},
            "edges": {str(k): v for k, v in self.edges.items()},
            "current_node": self.current_node,
            "branch_heads": self.branch_heads
        }
    
    def clear_graph(self):
        """Clear the session graph"""
        self.nodes.clear()
        self.edges.clear()
        self.current_node = 0
        self.branch_heads = [0]
        
        # Re-initialize root
        self.add_node(0, {"type": "root", "data": None})
    
    def get_interaction_path(self, start_node: int = 0) -> List[Dict]:
        """Get the path of interactions from start node"""
        return self.traverse(start_node)
    
    def get_current_path(self) -> List[Dict]:
        """Get the current interaction path"""
        return self.get_interaction_path(0)
    
    def create_branch(self, from_node: Optional[int] = None) -> int:
        """Create a new branch from a node"""
        branch_point = from_node if from_node is not None else self.current_node
        self.branch_heads.append(branch_point)
        return branch_point
    
    def switch_branch(self, branch_head: int) -> bool:
        """Switch to a different branch"""
        if branch_head in self.branch_heads:
            self.current_node = branch_head
            return True
        return False