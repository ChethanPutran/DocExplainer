import networkx as nx
from typing import Dict, List, Tuple
# import torch
import re
import plotly.graph_objects as go
from src.core.knowlege_modelling.user_model import UserState
from src.core.knowlege_modelling.base import Concept, ConceptRelationship, ConceptNode, ConceptNodeRelationship, ConceptGraph, GraphDelta
from src.core.document.document_processing import DocumentChunk, DocumentNode, DocumentTree
from src.core.document.document_cacher import DocumentCacher
from src.models.text import TextModels



class GraphUpdater:
    def __init__(self, base_graph: ConceptGraph, user_state: UserState):
        self.graph = base_graph
        self.user = user_state

    def apply_delta(self, delta: GraphDelta):
        # Add new concepts
        for concept in delta.new_concepts.values():
            self.graph.add_concept(concept)

        # Add new edges
        for edge in delta.new_edges:
            w = self.compute_weight(edge)
            self.graph.add_relationship(edge.concept1, edge.concept2, w)

        # Update existing edges
        for (u, v), dw in delta.edge_updates.items():
            self.graph.update_relationship(u, v, dw)

    def compute_weight(self, edge: ConceptNodeRelationship):
        cu = self.user.confidence.get(edge.concept1.primary_concept.name, 0.5)
        cv = self.user.confidence.get(edge.concept2.primary_concept.name, 0.5)
        return edge.relationship.strength * min(cu, cv)


class ConceptBuilder:
    """
    Builds knowledge graph from document concepts
    """
    GRAPH = "concept_graph"
    def __init__(self, text_model : TextModels) -> None:
        self.ner_model = text_model.get_ner_model()
        self.ner_regex = text_model.get_ner_regex()
        self.ner_llm = text_model.get_ner_llm()
        self.document_cacher = DocumentCacher()
        self.concept_embeddings = {}

    def _build_concept_graph(self, concepts: List[Tuple[Concept, List[Tuple[Concept,ConceptRelationship]]]], doc_sub_tree: DocumentTree) -> ConceptGraph:
        """
        Build concept graph from list of concepts
        """

        graph = ConceptGraph()
        # Create initial graph
        # self._build_initial_graph(concepts, [doc_sub_tree])
        
        # Add relationships
        # self._extract_relationships(filtered_concepts, all_text)
        for main_concept, list_tup in concepts:
            concept_node = ConceptNode(primary_concept=main_concept)
            for sec_concept, rel in list_tup:
                sec_concept_node = graph.get_concept(sec_concept.name)
                if sec_concept_node is None:
                    sec_concept_node = ConceptNode(primary_concept=sec_concept)
                    graph.add_concept(sec_concept_node)
                edge = ConceptNodeRelationship(
                    concept1=concept_node,
                    concept2=sec_concept_node,
                    relationship=rel)
                graph.add_relationship(concept_node, sec_concept_node, relationship=edge)
    
        return graph
    
    
    def extract_concepts_from_document(self, document_tree: DocumentTree, section:int = 0, paragraph = 0, context = 0) -> List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]:
        """
        Extract the concepts from the document in  the selected section and paragraph and build knowledge graph

        Used the cached knowlege for previous parts of the document if available

        Extract concepts using hybrid approach:
        1. NER for named entities
        2. Pattern matching for definitions
        3. LLM for abstract concepts
        """
        print("Extracting concepts from document...")

        cur_section = document_tree.get_section(section)
        # previous_sections = document_tree.get_previous_sections(section)

        # Check for cached concepts
        cached_knowledge = self.document_cacher.retrieve_document(section)
        
        # Add logioc for some context from previous sections/paragraphs if needed
        if cached_knowledge:
            return cached_knowledge

        texts = []
        for paragraph_node in cur_section:
            for sentence_node in paragraph_node.children:
                text = sentence_node.chunk.text
                texts.append(text)
        all_concepts = []

  
        # Method 1: NER for named entities
        ner_concepts = self.ner_model.extract_concepts(texts)

        # Method 2: Pattern-based extraction
        pattern_concepts = self.ner_regex.extract_concepts(texts)

        # Method 3: LLM-based extraction (if available)
        llm_concepts = []

        if self.ner_llm:
            llm_concepts = self.ner_llm.extract_concepts(texts)
        
        # Combine and deduplicate concepts
        all_concepts_t = list(set(ner_concepts + pattern_concepts + llm_concepts))
        all_concepts = self._build_concepts(all_concepts_t)

        # Filter and rank concepts
        filtered_concepts = self._filter_concepts(all_concepts)

        concept_relationships = self._extract_relationships(filtered_concepts, text=" ".join(texts))

        return concept_relationships

    def _build_concepts(self, text_concepts: List[str]) -> List[Concept]:
        concepts = []
        for t_concept in text_concepts:
            concept = Concept(name=t_concept, description=t_concept)
            concepts.append(concept)
        return concepts


    def _filter_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Filter and rank concepts by importance"""
        # Score concepts based on frequency and position
        scored_concepts = []
        
        for concept in concepts:
            concept_name = concept.name
            concept_text = concept.description
            n = len(concept_text)

            if n < 3 or n > 50:
                continue
            
            # Frequency in text
            frequency = concept_text.lower().count(concept_name)
            
            # Position of first occurrence (earlier = more important)
            first_pos = concept_text.lower().find(concept_name)
            position_score = 1.0 if first_pos == -1 else 1.0 / (1 + first_pos/1000)
            
            # Length score (medium length concepts are best)
            words = len(concept_text.split())
            length_score = 1.0 if 1 <= words <= 4 else 0.5
            
            # Combined score
            score = frequency * position_score * length_score
            
            if score > 0.5:  # Threshold
                concept.score = score
                concept.frequency = frequency
                concept.first_position = first_pos

                scored_concepts.append(concept)
        
        # Sort by score
        scored_concepts.sort(key=lambda x: x.score, reverse=True)

        # Return top 50 concepts
        return scored_concepts[:50]

    
    def _extract_relationships(self, concepts: List[Concept], text: str)-> List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]:
        """Extract relationships between concepts"""
        concepts_with_relationships = []

        # Co-occurrence relationships
        for i, concept1 in enumerate(concepts):
            concept_relations = []
            for concept2 in concepts[i+1:]:
                # Check if concepts appear together in sentences
                sentences = re.split(r'[.!?]+', text)
                co_occurrence_count = 0
                texts = []
                for sentence in sentences:
                    if (concept1.name.lower() in sentence.lower() and 
                        concept2.name.lower() in sentence.lower()):
                        co_occurrence_count += 1
                        texts.append(sentence.strip())
               
                if co_occurrence_count > 0:

                    weight = co_occurrence_count / len(sentences)

                    relationship = ConceptRelationship(
                        concept1,
                        concept2,
                        description=f"Co-occurs in {co_occurrence_count} sentences.",
                        attributes={'count': co_occurrence_count, 'weight': weight, 'texts': texts}
                    )
                    concept_relations.append((concept2, relationship))
            concepts_with_relationships.append((concept1, concept_relations))
        return concepts_with_relationships
        # # Hierarchical relationships (is_a, part_of)
        # hierarchical_patterns = [
        #     (r'(\w+)\s+(?:is a|is an|are)\s+(\w+)', 'is_a'),
        #     (r'(\w+)\s+(?:consists of|comprises|includes)\s+(\w+)', 'has_part'),
        #     (r'(\w+)\s+(?:is part of|is component of)\s+(\w+)', 'part_of'),
        # ]
        
        # for pattern, relation_type in hierarchical_patterns:
        #     matches = re.finditer(pattern, text, re.IGNORECASE)
        #     for match in matches:
        #         concept_a, concept_b = match.group(1), match.group(2)
                
        #         # Check if both concepts are in our graph
        #         if concept_a in self.graph and concept_b in self.graph:
        #             self.graph.add_edge(
        #                 concept_a, concept_b,
        #                 relation=relation_type,
        #                 weight=1.0
        #             )

    def visualize_graph(self, graph: nx.Graph, max_nodes=30):
        """Create interactive visualization of concept graph"""
        # Get top concepts by score
        nodes = list(graph.nodes(data=True))
        nodes.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        top_nodes = [n[0] for n in nodes[:max_nodes]]
        
        # Create subgraph
        subgraph = graph.subgraph(top_nodes)
        
        # Create Plotly visualization
        pos = nx.spring_layout(subgraph, seed=42)
        
        edge_traces = []
        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge[2].get('weight', 0.5) * 5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Score: {subgraph.nodes[node].get('score', 0):.2f}")
            node_size.append(subgraph.nodes[node].get('score', 1) * 50 + 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Importance')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Concept Knowledge Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    # def find_prerequisites(self, target_concept: str, user_knowledge: Set[str]) -> List[str]:
        """
        Find prerequisite concepts for a target concept
        """
        if target_concept not in self.graph:
            return []
        
        # BFS to find prerequisite chain
        visited = set()
        queue = [(target_concept, 0)]
        prerequisites = []
        
        while queue:
            concept, depth = queue.pop(0)
            if concept in visited:
                continue
            
            visited.add(concept)
            
            # Get incoming edges (concepts that this depends on)
            for predecessor in self.graph.predecessors(concept):
                edge_data = self.graph.get_edge_data(predecessor, concept)
                
                # Check if it's a prerequisite relationship
                if edge_data and edge_data.get('relation') in ['is_a', 'part_of', 'prerequisite']:
                    if predecessor not in user_knowledge:
                        prerequisites.append({
                            'concept': predecessor,
                            'depth': depth + 1,
                            'relation': edge_data.get('relation'),
                            'importance': self.graph.nodes[predecessor].get('score', 0)
                        })
                    
                    if depth < 3:  # Limit search depth
                        queue.append((predecessor, depth + 1))
        
        # Sort by importance and depth
        prerequisites.sort(key=lambda x: (-x['importance'], x['depth']))
        
        return prerequisites[:10]  # Return top 10
    

class DocumentChain:
    """
    Represents a chain of document sections or paragraphs.
    """
    def __init__(self):
        self.sections = None
        self.chain: List[GraphDelta] = []
        self.init_graph = ConceptGraph()
       

    def _append(self, delta):
        self.chain.append(delta)
    
    def get_concept_graph_upto(self, idx: int):
        return self.chain[:idx + 1]

    def get_document_context(self, check_point) -> Dict:
        context = {
            "text": "",
            "embeddings": []
        }
        text = ""
        for delta in self.chain[:check_point]:
            text += delta.data.text + "\n"
        context["text"] = text
        return context


class GraphStateManager:
    def __init__(self, text_models: TextModels):
        self.document_chain = DocumentChain()
        self.concept_builder = ConceptBuilder(text_models)
        self.graph = ConceptGraph()
        self.graph_updater = GraphUpdater(self.graph, UserState())

    def get_concept_graph_upto(self, idx: int):
        for delta in self.document_chain.get_concept_graph_upto(idx):
            self.graph_updater.apply_delta(delta)
        return self.graph
    
    def build_chain(self, document: DocumentTree):
        self.document = document
        self.sections = document.get_sections()

        # Chain building logic (temporal chain of sections)
        for section_id, section in enumerate(self.sections):
            # Extract concepts for the section
            concepts = self.concept_builder.extract_concepts_from_document(document, section=section_id)

            # Create delta
            delta = GraphDelta(section_id=section_id, data=section)
            delta.create(self.graph, concepts)
            
            # Append to document chain
            self.document_chain._append(delta)
    
    def get_document_context(self, check_point) -> Dict:
        return self.document_chain.get_document_context(check_point)
    
    
if __name__ == "__main__":
    document_text = """
    Intelligent Systems for Autonomous Decision Making
1. Introduction

Autonomous systems are increasingly deployed in real-world environments where they must perceive, reason, and act under uncertainty. These systems rely on a combination of sensing, learning, and control mechanisms to operate safely and efficiently. Recent advances in machine learning have enabled autonomous agents to handle complex scenarios that were previously considered intractable.

The core challenge in autonomous decision making lies in integrating heterogeneous information sources. Sensor data such as images, lidar measurements, and inertial signals must be processed jointly to produce a coherent understanding of the environment. This understanding is then used to generate actions that satisfy safety, efficiency, and robustness constraints.

2. System Architecture

A typical autonomous system consists of three primary modules: perception, decision making, and control. The perception module transforms raw sensor inputs into structured representations such as object lists, maps, or latent features. These representations provide the foundation for downstream reasoning.

The decision-making module operates on the perceived state of the environment. It may use rule-based logic, optimization techniques, or learned policies to determine appropriate actions. Reinforcement learning has emerged as a powerful framework for learning decision policies directly from interaction data.

3. Learning and Adaptation

Machine learning enables autonomous systems to adapt to new environments and changing conditions. Supervised learning is often used for perception tasks, while reinforcement learning is commonly applied to sequential decision problems. Unsupervised methods can assist in representation learning and anomaly detection.

Adaptation remains challenging due to limited data availability and safety constraints. Online learning techniques must balance exploration and exploitation while avoiding catastrophic failures. Incorporating prior knowledge and human feedback can significantly improve learning efficiency and system reliability.

4. Evaluation and Challenges

Evaluating autonomous systems requires carefully designed metrics that capture both performance and safety. Simulation environments are frequently used to test algorithms under controlled conditions before real-world deployment. However, simulation-to-reality gaps can limit the effectiveness of this approach.

Key challenges include robustness to distribution shifts, interpretability of learned models, and scalability to complex environments. Addressing these challenges is critical for the widespread adoption of autonomous technologies in safety-critical domains.

5. Conclusion

Autonomous decision-making systems combine perception, learning, and control to operate in uncertain environments. Advances in machine learning have significantly expanded their capabilities, but important challenges remain. Continued research in adaptive learning, robust evaluation, and human-in-the-loop systems is essential for building trustworthy autonomous agents.
    """
    sections = re.split(r'(?=^##\s+\d+\.\s+.+$)', document_text, flags=re.MULTILINE)

    chunk = DocumentChunk(text=document_text)
    root = DocumentNode(chunk)
    doc = DocumentTree("Root", root)
    doc.hierarchy = {
            'document': chunk,
            'sections': [DocumentChunk(text=s) for s in sections],
            'paragraphs': [
                DocumentChunk(text=p) for s in sections for p in re.split(r'\n\s*\n', s) if p.strip()
            ],
            'sentences': [
                DocumentChunk(text=sent) for s in sections for p in re.split(r'\n\s*\n', s) if p.strip() for sent in re.split(r'(?<=[.!?]) +', p) if sent.strip()       
            ]
        }
    
    manager = GraphStateManager(TextModels())
    manager.build_chain(doc)
    G = manager.get_concept_graph_upto(0)
    G.visualize()
