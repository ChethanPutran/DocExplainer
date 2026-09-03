from typing import Dict, List, Optional
from .node import Node


class PipelineContext:
    """Captures the nodes created during pipeline function execution."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._current_node: Optional[Node] = None

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def clear(self):
        self.nodes.clear()