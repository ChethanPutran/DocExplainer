import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional
from .builder import ConceptGraphBuilder

class GraphVisualizer:
    """Visualization utilities for concept graphs"""
    
    @staticmethod
    def visualize_matplotlib(graph: nx.DiGraph, figsize=(12, 12)):
        """Visualize graph using matplotlib"""
        if not graph.nodes():
            print("Graph is empty")
            return

        labels = {node: node for node in graph.nodes()}
        pos = nx.spring_layout(graph)

        plt.figure(figsize=figsize)
        nx.draw(graph, pos,
                labels=labels,
                with_labels=True,
                node_size=2000,
                node_color="lightblue",
                font_size=10,
                arrows=True)
        plt.title("Concept Graph")
        plt.show()

    @staticmethod
    def visualize_plotly(graph: nx.DiGraph, max_nodes: int = 30) -> go.Figure:
        """Create interactive Plotly visualization"""
        return ConceptGraphBuilder.visualize_graph(None, graph, max_nodes)

    @staticmethod
    def export_graphml(graph: nx.DiGraph, filepath: str):
        """Export graph to GraphML format"""
        nx.write_graphml(graph, filepath)
        print(f"Graph exported to {filepath}")

    @staticmethod
    def import_graphml(filepath: str) -> nx.DiGraph:
        """Import graph from GraphML format"""
        return nx.read_graphml(filepath)