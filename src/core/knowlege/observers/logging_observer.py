import logging
from typing import List
from .base import KnowledgeGraphObserver
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship

logger = logging.getLogger(__name__)

class LoggingObserver(KnowledgeGraphObserver):
    """Observer that logs knowledge graph events"""
    
    def on_concept_added(self, concept: Concept, section_id: int):
        """Log concept addition"""
        logger.info(f"Concept added: '{concept.name}' in section {section_id}")
    
    def on_relationship_added(self, relationship: ConceptRelationship):
        """Log relationship addition"""
        logger.info(f"Relationship added: '{relationship.concept1.name}' -[{relationship.relation}]-> '{relationship.concept2.name}'")
    
    def on_section_processed(self, section_id: int, concepts: List[Concept]):
        """Log section processing"""
        logger.info(f"Section {section_id} processed with {len(concepts)} concepts")