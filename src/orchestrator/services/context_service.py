from typing import Optional, Dict, Any
import logging

from src.core.memory.models.context import Context, SessionContext
from src.core.memory.managers.session_manager import SessionManager
from src.core.knowledge.models.graph import ConceptGraph

from .document_service import DocumentService
from .user_service import UserService


class ContextService:
    """Service for building context"""
    
    def __init__(self,
                 document_service: DocumentService,
                 user_service: UserService,
                 session_manager: SessionManager,
                 logger=None):
        self.document_service = document_service
        self.user_service = user_service
        self.session_manager = session_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def build_context(self,
                     user_id: str,
                     doc_id: str,
                     section_id: int = 0,
                     include_user_knowledge: bool = True,
                     include_session: bool = True,
                     include_document: bool = True,
                     include_graph: bool = True) -> Context:
        """Build comprehensive context"""
        self.logger.info(f"Building context for user {user_id}, doc {doc_id}, section {section_id}")
        
        # User knowledge
        user_knowledge = None
        if include_user_knowledge:
            user_knowledge = self.user_service.get_user_knowledge(user_id)
        
        # Session context
        session_context = None
        if include_session:
            session_context = self.session_manager.get_session_context()
        
        # Document context
        document_context = None
        if include_document:
            document_context = self.document_service.get_document_context(doc_id, section_id)
        
        # Concept graph
        concept_graph = None
        if include_graph:
            # This would need to be retrieved from graph state manager
            pass
        
        return Context(
            user_knowledge=user_knowledge,
            session_context=session_context,
            document_context=document_context,
            concept_graph=concept_graph or ConceptGraph()
        )
    
    def build_session_context(self, user_id: str) -> SessionContext:
        """Build session context only"""
        return self.session_manager.get_session_context()