from typing import Optional
from datetime import datetime

from .base import BasePipeline
from ..base.exceptions import DocumentNotFoundError
from ..models.requests import RegisterDocumentRequest, GetContextRequest
from ..models.responses import DocumentResponse, ContextResponse
from ..services.document_service import DocumentService
from ..services.context_service import ContextService


class DocumentPipeline(BasePipeline):
    """Pipeline for document operations"""
    
    def __init__(self, document_service: DocumentService,
                 context_service: Optional[ContextService] = None,
                 logger=None):
        super().__init__(logger)
        self.document_service = document_service
        self.context_service = context_service
    
    def _process(self, request: BaseRequest) -> DocumentResponse:
        """Process document request"""
        if isinstance(request, RegisterDocumentRequest):
            return self._process_registration(request)
        elif isinstance(request, GetContextRequest):
            return self._process_context(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")
    
    def _process_registration(self, request: RegisterDocumentRequest) -> DocumentResponse:
        """Process document registration"""
        self.logger.info(f"Registering document from path: {request.path}")
        
        try:
            doc_id = self.document_service.register_document(
                path=request.path,
                build_graph=request.build_graph,
                user_id=request.user_id
            )
            
            document = self.document_service.get_document(doc_id)
            
            return DocumentResponse(
                success=True,
                message="Document registered successfully",
                doc_id=doc_id,
                document=document,
                title=document.title if document else ""
            )
            
        except Exception as e:
            self.logger.error(f"Document registration failed: {e}")
            raise
    
    def _process_context(self, request: GetContextRequest) -> ContextResponse:
        """Process context retrieval"""
        self.logger.info(f"Getting context for doc {request.doc_id}, section {request.section_id}")
        
        # Check if document exists
        if not self.document_service.has_document(request.doc_id):
            raise DocumentNotFoundError(f"Document {request.doc_id} not found")
        
        if self.context_service:
            context = self.context_service.build_context(
                user_id=request.user_id,
                doc_id=request.doc_id,
                section_id=request.section_id,
                include_user_knowledge=request.include_user_knowledge,
                include_session=request.include_session,
                include_document=request.include_document,
                include_graph=request.include_graph
            )
            
            return ContextResponse(
                success=True,
                message="Context retrieved successfully",
                context=context,
                user_knowledge=context.user_knowledge.to_dict() if context.user_knowledge else {},
                session_data=context.session_context.to_dict() if context.session_context else {},
                document_context={"text": context.document_context.get("text", "")} if context.document_context else {},
                concept_graph_info={"node_count": len(context.concept_graph.graph.nodes)} if context.concept_graph else {}
            )
        else:
            return ContextResponse(
                success=False,
                message="Context service not available"
            )