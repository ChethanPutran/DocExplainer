from typing import List
from .base import KnowledgeGraphObserver
from ...knowledge.models.concept import Concept
from ...knowledge.models.relationship import ConceptNode, ConceptNodeRelationship
from ..repository import BaseKnowledgeStore

class KnowledgeStoreObserver(KnowledgeGraphObserver):
    """Observer that persists data to knowledge store"""
    
    def __init__(self, knowledge_store: BaseKnowledgeStore):
        self.knowledge_store = knowledge_store
    
    def on_concept_added(self, concept: ConceptNode, section_id: int):
        """Save concept to store"""
        self.knowledge_store.save_concept(concept)
    
    def on_relationship_added(self, relationship: ConceptNodeRelationship):
        """Save relationship to store"""
        # Convert to node relationship if needed
        pass
    
    def on_section_processed(self, section_id: int, concepts: List[ConceptNode]):
        """Update inverted index"""
        for concept in concepts:
            # Update index logic
            pass