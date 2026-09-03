from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from doc_explainer.core.common.dataclasses import ExplanationStyle
from doc_explainer.orchestrator.progress import ProgressReporter

from .base.exceptions import OrchestratorError

from .models.requests import (
    SummarizeRequest,
    ExplainRequest,
    AnswerRequest,
    RegisterDocumentRequest,
    GetContextRequest,
)

from .models.responses import (
    BaseResponse,
    AnswerResponse,
    DocumentResponse,
)

from .pipeline.document_pipeline import DocumentPipeline
from .pipeline.explanation_pipeline import ExplanationPipeline
from .pipeline.knowledge_pipeline import KnowledgePipeline
from .factories.pipeline_factory import PipelineFactory
from .config import OrchestratorConfig

logger = logging.getLogger(__name__)

class DocExplainerOrchestrator:
    """
    Main application orchestrator.

    Responsibilities:
        - Coordinate pipelines.
        - Construct requests.
        - Maintain application-level document metadata.
        - Propagate execution/progress context.

    The orchestrator is UI-independent.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
    ):
        self.config = config

        try:
            # ----------------------------------------------------------
            # Factory
            # ----------------------------------------------------------

            self.factory = PipelineFactory(
                config=self.config
            )

            # ----------------------------------------------------------
            # Pipelines
            # ----------------------------------------------------------

            self.document_pipeline: DocumentPipeline = (
                self.factory.create_document_pipeline()
            )

            self.explanation_pipeline: ExplanationPipeline = (
                self.factory.create_explanation_pipeline()
            )

            self.knowledge_pipeline: KnowledgePipeline = (
                self.factory.create_knowledge_pipeline()
            )

            # ----------------------------------------------------------
            # Application-level document metadata
            # ----------------------------------------------------------

            self.document_info: Dict[
                str,
                Dict[str, Any],
            ] = {}

        except Exception as e:
            logger.exception(
                "Error initializing orchestrator"
            )

            raise OrchestratorError(
                f"Failed to initialize orchestrator: {e}"
            ) from e

    # ==================================================================
    # DOCUMENT REGISTRATION
    # ==================================================================

    def register_document(
        self,
        path: str,
        build_graph: bool = True,
        user_id: Optional[str] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> BaseResponse:
        """
        Register a document.

        Progress is propagated through the pipeline into the
        DocumentService.

        This method is synchronous from the orchestrator's perspective.
        The GUI can execute it asynchronously using DocumentRegistrationWorker.
        """

        user_id = (
            user_id
            or self.config.default_user_id
        )

        logger.info(
            "Registering document: %s",
            path,
        )

        request = RegisterDocumentRequest(
            user_id=user_id,
            path=path,
            build_graph=build_graph,
        )

        try:
            response = self.document_pipeline.process(
                request,
                progress_reporter=progress_reporter,
                job_id=job_id,
            )

            # ----------------------------------------------------------
            # Registration successful
            # ----------------------------------------------------------

            if (
                response.success
                and isinstance(
                    response,
                    DocumentResponse,
                )
                and response.document_id
            ):
                document_id = str(
                    response.document_id
                )

                self.document_info[document_id] = {
                    "path": path,
                    "title": response.title,
                    "registered_at": datetime.now().isoformat(),
                }

                logger.info(
                    "Document registered successfully: %s",
                    path,
                )

            else:
                logger.error(
                    "Failed to register document: %s",
                    path,
                )

            return response

        except Exception:
            logger.exception(
                "Document registration failed: %s",
                path,
            )
            raise

    # ==================================================================
    # SUMMARIZATION
    # ==================================================================

    def summarize(
        self,
        doc_id: str,
        selected_text: str,
        section_id: int = 0,
        user_id: Optional[str] = None,
    ) -> BaseResponse:
        """Summarize selected text."""

        user_id = (
            user_id
            or self.config.default_user_id
        )

        request = SummarizeRequest(
            user_id=user_id,
            doc_id=doc_id,
            selected_text=selected_text,
            section_id=section_id,
        )

        return self.explanation_pipeline.process(
            request
        )

    # ==================================================================
    # EXPLANATION
    # ==================================================================

    def explain(
        self,
        doc_id: str,
        selected_text: str,
        section_id: int = 0,
        style: Optional[ExplanationStyle] = None,
        user_id: Optional[str] = None,
    ) -> BaseResponse:
        """Explain selected text."""

        user_id = (
            user_id
            or self.config.default_user_id
        )

        request = ExplainRequest(
            user_id=user_id,
            doc_id=doc_id,
            selected_text=selected_text,
            section_id=section_id,
            style=style,
        )

        return self.explanation_pipeline.process(
            request
        )

    # ==================================================================
    # QUESTION ANSWERING
    # ==================================================================

    def answer(
        self,
        doc_id: str,
        question: str,
        section_id: int = 0,
        user_id: Optional[str] = None,
    ) -> BaseResponse:
        """Answer a question about a document."""

        user_id = (
            user_id
            or self.config.default_user_id
        )

        request = AnswerRequest(
            user_id=user_id,
            doc_id=doc_id,
            question=question,
            section_id=section_id,
        )

        return self.explanation_pipeline.process(
            request
        )

    # ==================================================================
    # CONTEXT
    # ==================================================================

    def get_context(
        self,
        doc_id: str,
        section_id: int = 0,
        user_id: Optional[str] = None,
    ) -> BaseResponse:
        """Get context for a document and section."""

        user_id = (
            user_id
            or self.config.default_user_id
        )

        request = GetContextRequest(
            user_id=user_id,
            doc_id=doc_id,
            section_id=section_id,
        )

        return self.document_pipeline.process(
            request
        )

    # ==================================================================
    # DOCUMENT ACCESS
    # ==================================================================

    def get_section_id_at_position(
        self,
        doc_id: str,
        page: int,
        position: int,
    ) -> str:
        """Get section ID at a page and position."""

        document_service = (
            self.factory.create_document_service()
        )

        return document_service.get_section_id_at_position(
            doc_id,
            page,
            position,
        )

    def get_document(
        self,
        doc_id: str,
    ) -> Optional[Any]:
        """Get document by ID."""

        document_service = (
            self.factory.create_document_service()
        )

        return document_service.get_document(
            doc_id
        )

    # ==================================================================
    # KNOWLEDGE GRAPH
    # ==================================================================

    def analyze_prerequisites(
        self,
        concept_name: str,
    ) -> Dict[str, Any]:
        """Analyze prerequisites for a concept."""

        return self.knowledge_pipeline.analyze_prerequisites(
            concept_name
        )

    def generate_learning_path(
        self,
        target_concept: str,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate learning path for a concept."""

        return self.knowledge_pipeline.generate_learning_path(
            target_concept,
            max_depth,
        )

    def recommend_concepts(
        self,
        concept_name: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Recommend related concepts."""

        return self.knowledge_pipeline.recommend_concepts(
            concept_name,
            limit,
        )

    # ==================================================================
    # DOCUMENT METADATA
    # ==================================================================

    def get_document_info(
        self,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached document information."""

        return self.document_info.get(
            doc_id
        )

    def list_documents(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """List registered documents."""

        return self.document_info