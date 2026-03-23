from collections import defaultdict
from typing import List, Dict, Set

from src.core.user import UserManager
from ..repository import BaseKnowledgeStore
from ..graph.state_manager import GraphStateManager

class PrerequisiteAnalyzer:
    """Analyzes prerequisite relationships between concepts"""
    
    def __init__(self, 
                 graph_state_manager: GraphStateManager,
                 knowledge_store: BaseKnowledgeStore,
                 user_manager: UserManager):
        self.graph_state_manager = graph_state_manager
        self.knowledge_store = knowledge_store
        self.user_manager = user_manager

    def analyze_prerequisites(self, target_concept_name: str) -> Dict[str, List[str]]:
        """Analyze prerequisites for a target concept"""
        concept_graph = self.knowledge_store.graph
        user_knowledge = self.user_manager.get_user_knowledge()
        
        prerequisites = concept_graph.get_prerequisites(
            target_concept_name, 
            user_knowledge.confidence
        )
        
        target_concept = self.knowledge_store.get_concept_by_name(target_concept_name)
        dependents = []
        
        if target_concept:
            dependents = [
                c.name for c in concept_graph.get_dependents(target_concept.primary_concept)
            ]
        
        return {
            "prerequisites": [p["concept"] for p in prerequisites],
            "dependents": dependents,
            "prerequisite_details": prerequisites
        }
    
    def detect_section_prerequisites(self) -> Dict[int, Set[int]]:
        """Detect prerequisites between sections"""
        inverted_index = self.knowledge_store.get_inverted_index()
        section_prereq = defaultdict(set)

        for section in self.graph_state_manager.document.get_sections():
            for concept in section.concepts:
                # Skip if concept not indexed
                if concept.id not in inverted_index.index:
                    continue

                entry = inverted_index.index[concept.id]
                if entry.first_occurrence is None:
                    continue

                intro_section_id, _ = entry.first_occurrence

                # Rule 1: Concept introduced earlier
                if intro_section_id != section.id:
                    section_prereq[section.id].add(intro_section_id)

                # Rule 2: Dependency-based prerequisite
                concept_graph = self.knowledge_store.graph
                for dep in concept_graph.get_dependencies(concept):
                    if dep.id not in inverted_index.index:
                        continue

                    dep_entry = inverted_index.index[dep.id]
                    if dep_entry.first_occurrence is None:
                        continue

                    dep_intro_id, _ = dep_entry.first_occurrence
                    if dep_intro_id != section.id:
                        section_prereq[section.id].add(dep_intro_id)

        return dict(section_prereq)