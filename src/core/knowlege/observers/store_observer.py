from typing import List
from .base import KnowledgeGraphObserver
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship
from src.store.knowledge_store import BaseKnowledgeStore

class KnowledgeStoreObserver(KnowledgeGraphObserver):
    """Observer that persists data to knowledge store"""
    
    def __init__(self, knowledge_store: BaseKnowledgeStore):
        self.knowledge_store = knowledge_store
    
    def on_concept_added(self, concept: Concept, section_id: int):
        """Save concept to store"""
        self.knowledge_store.save_concept(concept)
    
    def on_relationship_added(self, relationship: ConceptRelationship):
        """Save relationship to store"""
        # Convert to node relationship if needed
        pass
    
    def on_section_processed(self, section_id: int, concepts: List[Concept]):
        """Update inverted index"""
        for concept in concepts:
            # Update index logic
            pass