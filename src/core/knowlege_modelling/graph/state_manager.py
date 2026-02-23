from typing import Dict
from collections import defaultdict
from src.core.document.document_structures import DocumentTree
from src.core.document.document_cacher import DocumentCacher
from src.core.knowlege_modelling.base import ConceptGraph, GraphDelta
from src.core.knowlege_modelling.user_model import UserKnowledgeState
from src.core.knowlege_modelling.extraction import ConceptExtractor,RelationshipExtractor
from src.models.text import TextModels

from .builder import ConceptBuilder
from .chain import DocumentChain
from .updater import GraphUpdater


class GraphStateManager:
    def __init__(self, text_models: TextModels, bkt_tracer=None):
        self.document_chain = DocumentChain()
        self.document_cacher = DocumentCacher()
        self.concept_extractor = ConceptExtractor(text_models)
        self.relation_extractor = RelationshipExtractor()
        self.concept_builder = ConceptBuilder(self.concept_extractor,self.relation_extractor,self.document_cacher)
        self.concept_graph = ConceptGraph()

        if bkt_tracer is None:
            # Backward compatibility for call sites that only passed text_models.
            from src.core.knowlege_modelling.knowledge_tracing import BayesianKnowledgeTracer

            bkt_tracer = BayesianKnowledgeTracer()

        self.bkt_tracer = bkt_tracer
        self.graph_updater = GraphUpdater(self.concept_graph, self._sync_user_state())

    def _sync_user_state(self) -> UserKnowledgeState:
        """
        Converts BKT knowledge probabilities into Graph confidence scores
        """
        user_knowlege_state = UserKnowledgeState()
        profile = self.bkt_tracer.get_user_knowledge_state()

        for concept, state in profile.knowledge_states.items():
            user_knowlege_state.confidence[concept.name] = state.p_knowledge
        return user_knowlege_state

    
    def get_concept_graph_upto(self, section_id: int):
        """
        Get the info till section
        """
        self.graph_updater.user = self._sync_user_state()

        for delta in self.document_chain.get_concept_graph_upto(section_id):
            self.graph_updater.apply_delta(delta)
        return self.concept_graph

    def build_chain(self, document: DocumentTree, concepts_per_para: int = 10):
        self.document = document
        self.sections = document.root.children

        # Extract the concepts in each section
        for section in self.sections.values():
            self.concept_builder.add_concepts_to_document(
                document, section.id, concepts_per_para
            )

            # Create how much info we get after reading this section
            delta = GraphDelta(section_id=section.id, data=section.chunk)
            delta.create(self.concept_graph, section.concept_relationships)

            # Add this delta to the chain
            self.document_chain._append(section.id, delta)

    def get_document_context(self, check_point) -> Dict:
        return self.document_chain.get_document_context(check_point)

    def detect_section_prerequisites(self):

        inverted_index = self.concept_extractor.get_inverted_index()
        section_prereq = defaultdict(set)

        for s2 in self.document.get_sections():

            for concept in s2.concepts:

                # ---- Skip if concept not indexed
                if concept.id not in inverted_index.index:
                    continue

                entry = inverted_index.index[concept.id]

                if entry.first_occurrence is None:
                    continue

                intro_section_id, _ = entry.first_occurrence

                # ---- Rule 1: Concept introduced earlier
                if intro_section_id != s2.id:
                    section_prereq[s2.id].add(intro_section_id)

                # ---- Rule 2: Dependency-based prerequisite
                for dep in self.concept_graph.get_dependencies(concept):

                    if dep.id not in inverted_index.index:
                        continue

                    dep_entry = inverted_index.index[dep.id]

                    if dep_entry.first_occurrence is None:
                        continue

                    dep_intro_id, _ = dep_entry.first_occurrence

                    if dep_intro_id != s2.id:
                        section_prereq[s2.id].add(dep_intro_id)

        return section_prereq