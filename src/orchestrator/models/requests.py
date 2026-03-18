from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseRequest:
    """Base request model"""
    user_id: str
    timestamp: Optional[str] = None


@dataclass
class SummarizeRequest(BaseRequest):
    """Request for summarization"""
    doc_id: str = ''
    selected_text: str = ''
    section_id: int = 0
    max_length: Optional[int] = None


@dataclass
class ExplainRequest(BaseRequest):
    """Request for explanation"""
    doc_id: str = ""
    selected_text: str = ""
    section_id: int = 0
    style: Optional[str] = None


@dataclass
class AnswerRequest(BaseRequest):
    """Request for question answering"""
    doc_id: str = ""
    question: str = ""
    section_id: int = 0


@dataclass
class RegisterDocumentRequest(BaseRequest):
    """Request for document registration"""
    path: str = ""
    build_graph: bool = True


@dataclass
class GetContextRequest(BaseRequest):
    """Request for context"""
    doc_id: str = ""
    section_id: int = 0
    include_user_knowledge: bool = True
    include_session: bool = True
    include_document: bool = True
    include_graph: bool = True