from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from doc_explainer.core.explanation_engine.models import Explanation
from doc_explainer.core.document.models import Document
from doc_explainer.core.memory.models.context import Context


@dataclass
class BaseResponse:
    """Base response model"""
    success: bool
    message: str = ""
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class SummarizeResponse(BaseResponse):
    """Response for summarization"""
    explanation: Optional[Explanation] = None
    summary: str = ""


@dataclass
class ExplainResponse(BaseResponse):
    """Response for explanation"""
    explanation: Optional[Explanation] = None


@dataclass
class AnswerResponse(BaseResponse):
    """Response for question answering"""
    explanation: Optional[Explanation] = None
    answer: str = ""


@dataclass
class DocumentResponse(BaseResponse):
    """Response for document operations"""
    document_id: Optional[str] = None
    document: Optional[Document] = None
    title: str = ""


@dataclass
class ContextResponse(BaseResponse):
    """Response for context retrieval"""
    context: Optional[Context] = None
    user_knowledge: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    document_context: Dict[str, Any] = field(default_factory=dict)
    concept_graph_info: Dict[str, Any] = field(default_factory=dict)