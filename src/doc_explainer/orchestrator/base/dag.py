from typing import List, Dict, Set
from .node import Node
from collections import deque


class DAG:
    """Represents a directed acyclic graph of Nodes."""
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.node_map = {n.id: n for n in nodes}
        self._validate()

    def _validate(self):
        # Check for cycles (simple DFS)
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            node = self.node_map[node_id]
            for dep_id in node.dependencies:
                if dep_id not in visited:
                    if dfs(dep_id):
                        return True
                elif dep_id in rec_stack:
                    raise ValueError(f"Cycle detected involving {node_id} and {dep_id}")
            rec_stack.remove(node_id)
            return False

        for nid in self.node_map:
            if nid not in visited:
                if dfs(nid):
                    raise ValueError("Cycle detected in DAG")

    def get_source_nodes(self) -> List[Node]:
        """Return nodes that have no dependencies (in‑degree zero)."""
        return [n for n in self.nodes if not n.dependencies]

    def get_sink_nodes(self) -> List[Node]:
        """Return nodes that are not depended on by any other node."""
        all_deps = set()
        for n in self.nodes:
            all_deps.update(n.dependencies)
        return [n for n in self.nodes if n.id not in all_deps]

    def topological_sort(self) -> List[Node]:
        """Return nodes in topological order."""
        indegree = {n.id: len(n.dependencies) for n in self.nodes}
        queue = deque([n.id for n in self.nodes if indegree[n.id] == 0])
        sorted_ids = []
        while queue:
            nid = queue.popleft()
            sorted_ids.append(nid)
            # find nodes that depend on this one
            for node in self.nodes:
                if nid in node.dependencies:
                    indegree[node.id] -= 1
                    if indegree[node.id] == 0:
                        queue.append(node.id)
        if len(sorted_ids) != len(self.nodes):
            raise ValueError("Cycle detected or incomplete graph")
        return [self.node_map[nid] for nid in sorted_ids]

    def get_dependents(self, node_id: str) -> List[Node]:
        """Return nodes that directly depend on this node."""
        return [n for n in self.nodes if node_id in n.dependencies]