from typing import Optional

from .base import BasePipeline

from ..base.exceptions import DocumentNotFoundError
from ..models.requests import (
    BaseRequest,
    RegisterDocumentRequest,
    GetContextRequest,
)
from ..models.responses import (
    DocumentResponse,
    ContextResponse,
)
from ..services.document_service import DocumentService
from ..services.context_service import ContextService
from ..progress import ProgressReporter


class DocumentPipeline(BasePipeline):
    """Pipeline for document operations."""

    def __init__(
        self,
        document_service: DocumentService,
        context_service: Optional[ContextService] = None,
        logger=None,
    ):
        super().__init__(logger)

        self.document_service = document_service
        self.context_service = context_service

    def _process(
        self,
        request: BaseRequest,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> DocumentResponse | ContextResponse:

        if isinstance(request, RegisterDocumentRequest):
            return self._process_registration(
                request=request,
                progress_reporter=progress_reporter,
                job_id=job_id,
            )

        elif isinstance(request, GetContextRequest):
            return self._process_context(request)

        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}"
            )

    def _process_registration(
        self,
        request: RegisterDocumentRequest,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> DocumentResponse:

        self.logger.info(
            f"Registering document from path: {request.path}"
        )

        try:
            doc_id = self.document_service.register_document(
                path=request.path,
                build_graph=request.build_graph,
                user_id=request.user_id,
                progress_reporter=progress_reporter,
                job_id=job_id,
            )

            document = self.document_service.get_document(doc_id)

            return DocumentResponse(
                success=True,
                message="Document registered successfully",
                document_id=doc_id,
                document=document,
                title=document.title if document else "",
            )

        except Exception as e:
            self.logger.error(
                f"Document registration failed: {e}"
            )
            raise

    def _process_context(
        self,
        request: GetContextRequest,
    ) -> ContextResponse:

        self.logger.info(
            f"Getting context for doc {request.doc_id}, "
            f"section {request.section_id}"
        )

        if not self.document_service.has_document(
            request.doc_id
        ):
            raise DocumentNotFoundError(
                f"Document {request.doc_id} not found"
            )

        if self.context_service:

            context = self.context_service.build_context(
                user_id=request.user_id,
                doc_id=request.doc_id,
                section_id=request.section_id,
                include_user_knowledge=request.include_user_knowledge,
                include_session=request.include_session,
                include_document=request.include_document,
                include_graph=request.include_graph,
            )

            return ContextResponse(
                success=True,
                message="Context retrieved successfully",
                context=context,
                user_knowledge=(
                    context.user_knowledge.to_dict()
                    if context.user_knowledge
                    else {}
                ),
                session_data=(
                    context.session_context.to_dict()
                    if context.session_context
                    else {}
                ),
                document_context=(
                    {
                        "text": context.document_context.get(
                            "text", ""
                        )
                    }
                    if context.document_context
                    else {}
                ),
                concept_graph_info=(
                    {
                        "node_count": len(
                            context.concept_graph.graph.nodes
                        )
                    }
                    if context.concept_graph
                    else {}
                ),
            )

        return ContextResponse(
            success=False,
            message="Context service not available",
        )