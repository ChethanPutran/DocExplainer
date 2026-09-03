from typing import Optional
from datetime import datetime

from .base import BasePipeline
from ..base.exceptions import DocumentNotFoundError
from ..models.requests import (
    SummarizeRequest,
    ExplainRequest,
    AnswerRequest,
    BaseRequest,
)
from ..models.responses import (
    SummarizeResponse,
    ExplainResponse,
    AnswerResponse,
    BaseResponse,
)
from ..services.document_service import DocumentService
from ..services.context_service import ContextService
from ...core.explanation_engine import AdaptiveExplainer
from ...core.user import UserManager
from ...core.memory.managers.memory_manager import MemoryManager
from ...core.memory.managers.session_manager import SessionManager
from ..progress import ProgressReporter


class ExplanationPipeline(BasePipeline):
    """Pipeline for explanation operations."""

    def __init__(
        self,
        document_service: DocumentService,
        context_service: ContextService,
        explainer: AdaptiveExplainer,
        user_manager: UserManager,
        memory_manager: MemoryManager,
        session_manager: SessionManager,
        logger=None,
    ):
        super().__init__(logger)

        self.document_service = document_service
        self.context_service = context_service
        self.explainer = explainer
        self.user_manager = user_manager
        self.memory_manager = memory_manager
        self.session_manager = session_manager

    def _process(
        self,
        request: BaseRequest,
        progress_reporter: Optional[ProgressReporter] = None,
        job_id: Optional[str] = None,
    ) -> BaseResponse:

        if isinstance(request, SummarizeRequest):
            return self._process_summarize(request)

        elif isinstance(request, ExplainRequest):
            return self._process_explain(request)

        elif isinstance(request, AnswerRequest):
            return self._process_answer(request)

        else:
            raise ValueError(
                f"Unsupported request type: {type(request)}"
            )

    def _process_summarize(
        self,
        request: SummarizeRequest,
    ) -> SummarizeResponse:

        self.logger.info(
            f"Summarizing text for doc {request.doc_id}"
        )

        if not self.document_service.has_document(
            request.doc_id
        ):
            raise DocumentNotFoundError(
                f"Document {request.doc_id} not found"
            )

        self.session_manager.handle_interaction(
            "summarize",
            request.selected_text,
        )

        context = self.context_service.build_context(
            user_id=request.user_id,
            doc_id=request.doc_id,
            section_id=request.section_id,
        )

        start_time = datetime.now()

        explanation = self.explainer.summarize(
            text=request.selected_text,
            context=context,
        )

        for concept in explanation.unknown_concepts_explained:
            self.user_manager.update_user_knowledge({
                "concept": concept,
                "correct": True,
            })

        self.session_manager.handle_interaction(
            "summarization_response",
            explanation.explanation,
        )

        self.memory_manager.handle_event(
            "summarization",
            {
                "text": request.selected_text,
                "summary": explanation.explanation,
            },
        )

        return SummarizeResponse(
            success=True,
            message="Summarization completed",
            explanation=explanation,
            summary=explanation.explanation,
        )

    def _process_explain(
        self,
        request: ExplainRequest,
    ) -> ExplainResponse:

        self.logger.info(
            f"Explaining text for doc {request.doc_id}"
        )

        if not self.document_service.has_document(
            request.doc_id
        ):
            raise DocumentNotFoundError(
                f"Document {request.doc_id} not found"
            )

        self.session_manager.handle_interaction(
            "explain",
            request.selected_text,
        )

        context = self.context_service.build_context(
            user_id=request.user_id,
            doc_id=request.doc_id,
            section_id=request.section_id,
        )

        if request.style:
            self.explainer.set_explanation_style(
                request.style
            )

        explanation = self.explainer.explain(
            text=request.selected_text,
            context=context,
        )

        for concept in explanation.unknown_concepts_explained:
            self.user_manager.update_user_knowledge({
                "concept": concept,
                "correct": True,
            })

        self.user_manager.update_user_knowledge({
            "text": request.selected_text,
            "explanation": explanation.explanation,
        })

        self.memory_manager.handle_event(
            "explanation",
            {
                "text": request.selected_text,
                "explanation": explanation.explanation,
            },
        )

        self.session_manager.handle_interaction(
            "explain_response",
            explanation.explanation,
        )

        return ExplainResponse(
            success=True,
            message="Explanation completed",
            explanation=explanation,
        )

    def _process_answer(
        self,
        request: AnswerRequest,
    ) -> AnswerResponse:

        self.logger.info(
            f"Answering question for doc {request.doc_id}"
        )

        if not self.document_service.has_document(
            request.doc_id
        ):
            raise DocumentNotFoundError(
                f"Document {request.doc_id} not found"
            )

        self.session_manager.handle_interaction(
            "answer_question",
            request.question,
        )

        context = self.context_service.build_context(
            user_id=request.user_id,
            doc_id=request.doc_id,
            section_id=request.section_id,
        )

        explanation = self.explainer.ask(
            question=request.question,
            context=context,
        )

        for concept in explanation.unknown_concepts_explained:
            self.user_manager.update_user_knowledge({
                "concept": concept,
                "correct": True,
            })

        self.user_manager.update_user_knowledge({
            "question": request.question,
            "answer": explanation.explanation,
        })

        self.memory_manager.handle_event(
            "question_answer",
            {
                "question": request.question,
                "answer": explanation.explanation,
            },
        )

        self.session_manager.handle_interaction(
            "answer_response",
            explanation.explanation,
        )

        return AnswerResponse(
            success=True,
            message="Question answered",
            explanation=explanation,
            answer=explanation.explanation,
        )