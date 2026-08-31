import logging
from typing import List
from .base import KnowledgeGraphObserver
from ...knowledge.models.relationship import ConceptNode, ConceptNodeRelationship

logger = logging.getLogger(__name__)

class LoggingObserver(KnowledgeGraphObserver):
    """Observer that logs knowledge graph events"""
    
    def on_concept_added(self, concept: ConceptNode, section_id: int):
        """Log concept addition"""
        logger.info(f"Concept added: '{concept.primary_concept.name}' in section {section_id}")
    
    def on_relationship_added(self, relationship: ConceptNodeRelationship):
        """Log relationship addition"""
        logger.info(f"Relationship added: '{relationship.concept1.primary_concept.name}' -[{relationship.relationship}]-> '{relationship.concept2.primary_concept.name}'")
    
    def on_section_processed(self, section_id: int, concepts: List[ConceptNode]):
        """Log section processing"""
        logger.info(f"Section {section_id} processed with {len(concepts)} concepts")