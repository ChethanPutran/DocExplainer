from typing import List
from .base import KnowledgeGraphObserver
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship
from src.core.knowledge.graph.chain import DocumentChain
from src.core.knowledge.models.delta import GraphDelta

class ChainObserver(KnowledgeGraphObserver):
    """Observer that updates the document chain"""
    
    def __init__(self, document_chain: DocumentChain):
        self.document_chain = document_chain
        self.current_delta = None
    
    def on_section_processed(self, section_id: int, concepts: List[Concept]):
        """Create and add delta to chain"""
        if self.current_delta:
            self.document_chain.add(section_id, self.current_delta)
            self.current_delta = None
    
    def set_current_delta(self, delta: GraphDelta):
        """Set the current delta being built"""
        self.current_delta = delta