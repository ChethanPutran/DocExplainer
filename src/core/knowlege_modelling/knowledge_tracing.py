import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
import plotly.graph_objects as go
from src.core.document.document_processing import DocumentTree
from src.core.document.document_cacher import DocumentCacher


class ConceptTree:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}

    def add_concept(self, concept: str, embedding: torch.Tensor):
        self.graph.add_node(concept)
        self.concept_embeddings[concept] = embedding

    def add_relationship(self, concept1: str, concept2: str, weight: float):
        self.graph.add_edge(concept1, concept2, weight=weight)

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10)
        plt.show()

class Concept:
    def __init__(self, 
                 name: str,
                 description: str = "", score: float = 0.0,
                 frequency: int = 0,
                 first_pos: int = -1,
                 attributes: Dict | None = None
                    ):
        self.name = name
        self.score = score
        self.frequency = frequency
        self.first_position = first_pos

        self.description = description
        self.attributes = attributes if attributes is not None else {}

class ConceptRelationship:
    def __init__(self, 
                 concept1: Concept,
                 concept2: Concept,
                 description: str = "",
                 attributes: Dict | None = None
                 ):
        self.concept1 = concept1
        self.concept2 = concept2
        self.description = description
        self.attributes = attributes if attributes is not None else {}

class ConceptNode:
    def __init__(self, primary_concept: Concept,
                  embedding: torch.Tensor | None = None
                  ):
        self.primary_concept = primary_concept
        self.embedding = embedding # For future use GNN

class ConceptGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}

    def add_concept(self, concept: ConceptNode):
        self.graph.add_node(concept)

    def add_relationship(self, concept1: ConceptNode, concept2: ConceptNode, relationship: ConceptRelationship):
        self.graph.add_edge(concept1, concept2, relationship=relationship)

    def get_concept(self, concept_name: str) -> ConceptNode | None:
        for concept in self.graph.nodes:
            if concept.primary_concept.name == concept_name:
                return concept
        return None
    
    def visualize(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color = "lightblue", font_size=10)
        plt.show()

class ConceptGraphBuilder:
    """
    Builds knowledge graph from document concepts
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.document_cacher = DocumentCacher()
        self.concept_embeddings = {}
        
        # Load NER model for concept extraction
        # self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        # self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "elastic/distilbert-base-cased-finetuned-conll03-english"
        )


    def build_concept_graph(self, concepts: List[Tuple[Concept, List[Tuple[Concept,ConceptRelationship]]]], doc_sub_tree: DocumentTree) -> ConceptGraph:
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
                graph.add_relationship(concept_node, sec_concept_node, relationship=rel)
    
        return graph
    
    
    def extract_concepts_from_document(self, document_tree: DocumentTree, section = "*", paragraph = "*", context = "*") -> ConceptGraph:
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
        previous_sections = document_tree.get_previous_sections(section)

        cached_knowledge = self.document_cacher.retrieve_document(section)
        cached_concepts = None
        if cached_knowledge:
            if 'graph' in cached_knowledge:
                return cached_knowledge['graph']
            cached_concepts = cached_knowledge['concepts']

        texts = []
        for paragraph_node in cur_section:
            for sentence_node in paragraph_node.children:
                text = sentence_node.chunk.text
                texts.append(text)
        all_concepts = []

        # Use cached concepts if available
        if cached_concepts:
            all_concepts.extend(cached_concepts)
        else:
            # Method 1: NER for named entities
            ner_concepts = self._extract_with_ner(texts)

            # Method 2: Pattern-based extraction
            pattern_concepts = self._extract_with_patterns(texts)

            # Method 3: LLM-based extraction (if available)
            llm_concepts = []
            if self.llm:
                llm_concepts = self._extract_with_llm(texts)

            # Combine and deduplicate concepts
            all_concepts = list(set(ner_concepts + pattern_concepts + llm_concepts))

        # Filter and rank concepts
        filtered_concepts = self._filter_concepts(all_concepts)

        concept_relationships = self._extract_relationships(filtered_concepts, text=" ".join(texts))

        return self.build_concept_graph(concept_relationships, document_tree)

    def _extract_with_ner(self, text:  List[str]) -> List[Concept]:
        """Extract concepts using Named Entity Recognition"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        concepts = []
        current_entity = []
        current_label = None
        
        for token, prediction in zip(tokens, predictions[0]):
            label = self.ner_model.config.id2label[prediction.item()]
            
            if label.startswith("B-"):
                if current_entity:
                    concepts.append(" ".join(current_entity))
                current_entity = [token.replace("##", "")]
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token.replace("##", ""))
            else:
                if current_entity:
                    concepts.append(" ".join(current_entity))
                current_entity = []
                current_label = None
        
        if current_entity:
            concepts.append(" ".join(current_entity))
        
        return list(set([c for c in concepts if len(c) > 2]))
    
    def _extract_with_patterns(self, text:  List[str]) -> List[Concept]:
        """Extract concepts using linguistic patterns"""
        patterns = [
            r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized phrases
            r'(?:"([^"]+)"\s+(?:is|are|means|refers to))',  # Quoted definitions
            r'(?:\b(?:the|a|an)\s+([A-Za-z-]+\s+(?:of|in|for)\s+[A-Za-z-]+))',  # Noun phrases
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                concept = match.group(1) if match.groups() else match.group(0)
                if concept and len(concept.split()) <= 5:  # Limit to 5 words
                    concepts.append(concept.strip())
        
        return list(set(concepts))
    
    def _extract_with_llm(self, text:  List[str]) -> List[Concept]:
        """Extract concepts using LLM"""
        prompt = f"""
        Extract the key concepts from the following text. 
        Return only the concepts as a comma-separated list.
        
        Text: {text[:2000]}
        
        Concepts:
        """
        
        try:
            response = self.llm.generate(prompt)
            concepts = [c.strip() for c in response.split(',')]
            return concepts
        except:
            return []

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