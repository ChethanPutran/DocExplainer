from __future__ import annotations
from typing import Optional, Dict, Any
import logging

from src.core.document import DocumentManager, Document, DocumentTree
from src.core.knowledge import GraphStateManager

from ..base.exceptions import DocumentNotFoundError


class DocumentService:
    """Service for document operations"""
    
    def __init__(self, 
                 document_manager: DocumentManager,
                 graph_state_manager: GraphStateManager,
                 logger=None):
        self.document_manager = document_manager
        self.graph_state_manager = graph_state_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.document_trees: Dict[str, DocumentTree] = {}
    
    def register_document(self, path: str, build_graph: bool = True,
                         user_id: Optional[str] = None) -> str:
        """Register a document"""
        self.logger.info(f"Registering document from path: {path}")
        
        # Load document
        doc_id = self.document_manager.load_document(path)
        
        # Build document tree
        document_tree = self.document_manager.build_document_tree(doc_id)
        self.document_trees[doc_id] = document_tree
        
        # Build knowledge graph if requested
        if build_graph:
            self.logger.info("Building knowledge graph...")
            self.graph_state_manager.build_chain(document_tree)
            self.logger.info("Knowledge graph built")
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        try:
            return self.document_manager.get_document(doc_id)
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def get_document_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Get document tree by ID"""
        return self.document_trees.get(doc_id)
    
    def has_document(self, doc_id: str) -> bool:
        """Check if document exists"""
        try:
            return self.document_manager.has_document(doc_id)
        except:
            return False
    
    def get_section_id_at_position(self, doc_id: str, page: int, position: int) -> int:
        """Get section ID at page and position"""
        doc = self.get_document(doc_id)
        if not doc:
            return -1
        
        for section in doc.sections:
            for para in section.paragraphs:
                if position == 0:
                    if para.page == page:
                        return int(section.id)
                elif para.page == page and (para.start <= position <= para.end):
                    return int(section.id)
        
        return -1
    
    def get_document_context(self, doc_id: str, section_id: int) -> Dict[str, Any]:
        """Get document context up to section"""
        return self.graph_state_manager.get_document_context(str(section_id))