from __future__ import annotations
from typing import List, TYPE_CHECKING
import networkx as nx
import plotly.graph_objects as go

if TYPE_CHECKING:
    from src.core.document import BaseDocumentCache, DocumentTree
    
from ..models import Concept, ConceptRelationship, ConceptNode, ConceptNodeRelationship,  ConceptGraph
from ..extraction import ConceptExtractor, LLMRelationshipExtractor,  StatisticalRelationshipExtractor

class ConceptGraphBuilder:
    """Builds knowledge graph from document concepts"""
    
    GRAPH = "concept_graph"

    def __init__(self, 
                 concept_extractor: ConceptExtractor,
                 llm_relationship_extractor: LLMRelationshipExtractor,
                 statistical_relationship_extractor: StatisticalRelationshipExtractor,
                 document_cacher: BaseDocumentCache) -> None:
        self.concept_extractor = concept_extractor
        self.llm_relationship_extractor = llm_relationship_extractor
        self.statistical_relationship_extractor = statistical_relationship_extractor
        self.document_cacher = document_cacher
        self.graph = ConceptGraph()

    def _validate_no_cycles(self) -> bool:
        """Check if graph has cycles"""
        try:
            cycle = nx.find_cycle(self.graph.graph)
            print("Cycle detected:", cycle)
            return False
        except nx.NetworkXNoCycle:
            return True

    def _build_concept_graph(self, concepts: List[Concept], 
                            relationships: List[ConceptRelationship]):
        """Construct a ConceptGraph from concepts and relationships"""
        # Add concept nodes
        for concept in concepts:
            if not self.graph.has_concept(concept.name):
                concept_node = self.graph.get_concept(concept.name) or ConceptNode(primary_concept=concept)
                self.graph.add_concept_node(concept_node)

        # Add relationships
        for relationship in relationships:
            concept1 = relationship.concept1
            concept2 = relationship.concept2

            concept1_node = self.graph.get_concept(concept1.name)
            concept2_node = self.graph.get_concept(concept2.name)

            if concept1_node and concept2_node:
                edge = ConceptNodeRelationship(
                    concept1=concept1_node, 
                    concept2=concept2_node, 
                    relationship=relationship
                )
                self.graph.add_relationship(concept1_node, concept2_node, relationship=edge)

    def visualize_graph(self, graph: nx.DiGraph, max_nodes=30) -> go.Figure:
        """Create interactive Plotly visualization"""
        if not graph.nodes:
            print("Graph is empty, nothing to visualize.")
            return go.Figure()

        nodes_with_data = list(graph.nodes(data=True))
        nodes_with_data.sort(
            key=lambda x: x[1]["data"].primary_concept.score, reverse=True
        )

        top_node_names = [n[0] for n in nodes_with_data[:max_nodes]]
        subgraph = graph.subgraph(top_node_names)
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

        edge_traces = []
        for u, v, data in subgraph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            rel = data.get("relationship")
            weight = rel.relationship.strength if rel else 0.5

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=max(1, weight * 3), color="#A9A9A9"),
                hoverinfo="none",
                mode="lines",
            )
            edge_traces.append(edge_trace)

        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []

        for node_name in subgraph.nodes():
            x, y = pos[node_name]
            node_x.append(x)
            node_y.append(y)

            node_obj = subgraph.nodes[node_name]["data"]
            score = node_obj.primary_concept.score
            freq = node_obj.primary_concept.frequency

            node_text.append(
                f"<b>Concept:</b> {node_name}<br>"
                f"<b>Importance Score:</b> {score:.2f}<br>"
                f"<b>Frequency:</b> {freq}"
            )

            node_size.append(max(15, score * 60))
            node_color.append(score)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[n for n in subgraph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Concept Importance",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="<br>Document Knowledge Landscape",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            ),
        )

        return fig

    def add_concepts_to_document(self, document_tree: DocumentTree, 
                                section_id: str, top: int = 5):
        """Extract concepts from a section and store in DocumentTree"""
        print("Extracting concepts from document...")

        cur_section = document_tree.get_section(section_id)
        
        assert cur_section is not None, f"Section with ID {section_id} not found in document tree"

        section_concepts = []
        section_concept_relations = []

        pre_section_summary = document_tree.get_previous_sections_summaries(section_id)
        pre_section_summary = "".join(pre_section_summary)
        para_summary = ""


        for paragraph_node in cur_section.children.values():
            texts = []
            for sentence_node in paragraph_node.children.values():
                text = sentence_node.chunk.text
                texts.append(text)
            text = "".join(texts)

            para_summary += paragraph_node.chunk.summary
            context = pre_section_summary + para_summary

            # Extract concepts
            concepts = self.concept_extractor.extract([text], context, section_id, paragraph_node.id)
            
            # Extract relationships
            stat_relations = self.statistical_relationship_extractor.extract(concepts, text, context)
            llm_relations = self.llm_relationship_extractor.extract(concepts, text, context)
            relationships = llm_relations + stat_relations

            # Add to paragraph
            paragraph_node.concepts = concepts
            paragraph_node.concept_relationships = relationships

            # Store for section
            section_concepts.extend(concepts)
            section_concept_relations.extend(relationships)

        # Store at section level
        cur_section.concepts = section_concepts
        cur_section.concept_relationships = section_concept_relations
        
        # Build graph
        self._build_concept_graph(section_concepts, section_concept_relations)