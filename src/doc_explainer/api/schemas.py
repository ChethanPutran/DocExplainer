"""Wire models, shared by the REST routes and the JSON-RPC methods.

Request models are strict (unknown fields are rejected); response models mirror what clients get.
The SDK keeps its own copy of the response models, so it never imports the backend.
"""
from __future__ import annotations
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from doc_explainer.core.common.enums import ExplanationDepth, ExplanationLevel


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ----------------------------------------------------------------------------- requests
class StyleBody(_Request):
    level: ExplanationLevel
    depth: ExplanationDepth
    description: str = ""
    examples: Optional[List[str]] = None


class ExplainBody(_Request):
    selected_text: str = Field(min_length=1, max_length=20000)
    section_id: int = Field(default=0, ge=0)
    style: Optional[StyleBody] = None
    session_id: Optional[str] = Field(default=None, max_length=64)    # record the exchange in this session


class SummarizeBody(_Request):
    selected_text: str = Field(min_length=1, max_length=20000)
    section_id: int = Field(default=0, ge=0)
    session_id: Optional[str] = Field(default=None, max_length=64)


class AskBody(_Request):
    question: str = Field(min_length=1, max_length=4000)
    section_id: int = Field(default=0, ge=0)
    session_id: Optional[str] = Field(default=None, max_length=64)


class ChatBody(_Request):
    """One turn of the conversation. ``kind``: what the reader wants done with ``text``."""
    kind: Literal["explain", "summarize", "ask"] = "ask"
    text: str = Field(min_length=1, max_length=20000)
    section_id: int = Field(default=0, ge=0)
    session_id: Optional[str] = Field(default=None, max_length=64)     # None = the session used last (or a new one)
    level: Optional[ExplanationLevel] = None


class SessionCreateBody(_Request):
    title: str = Field(default="", max_length=200)


class SessionRenameBody(_Request):
    title: str = Field(min_length=1, max_length=200)


class NoteBody(_Request):
    text: str = Field(min_length=1, max_length=20000)
    kind: Literal["note", "summary"] = "note"


# Parameter models for JSON-RPC: the REST path/query parameters become fields.
class RegisterParams(_Request):
    filename: str = Field(min_length=1, max_length=255)
    content_b64: str = Field(min_length=1)
    build_graph: bool = True


class DocIdParams(_Request):
    doc_id: str = Field(min_length=1, max_length=64)


class SectionAtParams(DocIdParams):
    page: int = Field(ge=0)
    position: int = Field(ge=0)


class ExplainParams(DocIdParams, ExplainBody):
    pass


class ChatParams(DocIdParams, ChatBody):
    pass


class SessionIdParams(DocIdParams):
    session_id: str = Field(min_length=1, max_length=64)


class SessionCreateParams(DocIdParams, SessionCreateBody):
    pass


class SessionRenameParams(SessionIdParams, SessionRenameBody):
    pass


class NoteParams(SessionIdParams, NoteBody):
    pass


class NoteIdParams(SessionIdParams):
    note_id: str = Field(min_length=1, max_length=64)


class NoteEditParams(NoteIdParams):
    text: str = Field(min_length=1, max_length=20000)


class SummarizeParams(DocIdParams, SummarizeBody):
    pass


class AskParams(DocIdParams, AskBody):
    pass


class ContextParams(DocIdParams):
    section_id: int = Field(default=0, ge=0)


class ConceptParams(DocIdParams):
    name: str = Field(min_length=1, max_length=200)


class PrerequisiteParams(ConceptParams):
    max_depth: int = Field(default=3, ge=1, le=10)


class LearningPathParams(ConceptParams):
    pass


class RecommendParams(ConceptParams):
    limit: int = Field(default=5, ge=1, le=50)


class KnowledgeParams(DocIdParams):
    upto: Optional[int] = Field(default=None, ge=0)        # reading-order position; None = the whole document


class SectionParams(DocIdParams):
    section_id: int = Field(ge=0)


class QuizBody(_Request):
    section_id: Optional[int] = Field(default=None, ge=0)
    count: int = Field(default=5, ge=1, le=10)


class QuizParams(DocIdParams, QuizBody):
    pass


class AnswerBody(_Request):
    choice: int = Field(ge=0, le=9)


class AnswerParams(DocIdParams, AnswerBody):
    question_id: str = Field(min_length=1, max_length=64)


class DiagnosticBody(_Request):
    level: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced|expert)$")
    known_topics: List[str] = Field(default_factory=list, max_length=30)
    questions: int = Field(default=6, ge=1, le=12)


class DiagnosticParams(DocIdParams, DiagnosticBody):
    pass


class FeedbackBody(_Request):
    verdict: str = Field(pattern="^(too_easy|too_hard|just_right)$")
    concept_ids: List[str] = Field(min_length=1, max_length=20)


class FeedbackParams(DocIdParams, FeedbackBody):
    pass


class ProgressBody(_Request):
    section_id: int = Field(ge=0)


class ProgressParams(DocIdParams, ProgressBody):
    pass


class JobIdParams(_Request):
    job_id: str = Field(min_length=1, max_length=64)


class NoParams(_Request):
    pass


# ----------------------------------------------------------------------------- responses
class ExplanationStyleOut(_Response):
    level: str
    depth: str
    description: str = ""
    examples: Optional[List[Any]] = None


class ResourceSuggestionOut(_Response):
    concept: str
    resource_type: str
    difficulty: str


class ExplanationMetadataOut(_Response):
    estimated_complexity: float = 0.0
    user_knowledge_matched: bool = False
    gap_bridging: bool = False
    processing_time_ms: float = 0.0


class ResourceOut(_Response):
    title: str
    url: str
    type: str
    description: str = ""
    difficulty: Any = None
    source: str = ""


class ExplanationOut(_Response):
    explanation: str
    style: Optional[ExplanationStyleOut] = None
    context_used: dict = Field(default_factory=dict)
    known_concepts_used: List[str] = Field(default_factory=list)
    unknown_concepts_explained: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    suggested_resources: List[ResourceSuggestionOut] = Field(default_factory=list)
    metadata: Optional[ExplanationMetadataOut] = None
    resources: List[ResourceOut] = Field(default_factory=list)


class ExplanationResult(_Response):
    text: str
    explanation: ExplanationOut
    processing_time_ms: float = 0.0


class ChatResult(ExplanationResult):
    session_id: str = ""
    message_id: int = 0


class DocumentInfo(_Response):
    doc_id: str
    title: str
    filename: str
    size_bytes: int
    uploaded_at: str
    knowledge: str = "none"        # ready | partial | failed | none: whether the knowledge graph was built
    sections: int = 0
    concepts: int = 0


class JobError(_Response):
    code: str
    message: str


class JobInfo(_Response):
    """A document registration running in the background.

    state: queued -> running -> succeeded | failed. ``document`` is set on success, ``error`` on
    failure. ``queue_position`` is 1-based while queued. ``reused`` means identical content was
    already registered, so nothing was re-processed.
    """
    job_id: str
    kind: str = "register_document"
    state: str
    filename: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    queue_position: Optional[int] = None
    reused: bool = False
    progress: float = 0.0          # 0..1 across parsing, indexing and building the knowledge graph
    stage: str = ""                # what it is doing now
    document: Optional["DocumentInfo"] = None
    error: Optional[JobError] = None


class SessionOut(_Response):
    id: str
    doc_id: str
    title: str
    created_at: float
    updated_at: float
    messages: int = 0
    notes: int = 0


class SessionList(_Response):
    sessions: List[SessionOut]


class MessageOut(_Response):
    id: int
    role: str                      # user | assistant
    text: str
    kind: str                      # ask | explain | summarize
    created_at: float
    meta: dict = Field(default_factory=dict)     # follow_ups, section_id, concept_ids, resources


class NoteOut(_Response):
    id: str
    kind: str                      # note | summary
    text: str
    created_at: float


class SessionDetail(_Response):
    session: SessionOut
    messages: List[MessageOut]
    notes: List[NoteOut]


class JobList(_Response):
    jobs: List[JobInfo]


class DocumentList(_Response):
    documents: List[DocumentInfo]


class SectionRef(_Response):
    section_id: int


class ContextOut(_Response):
    user_knowledge: dict = Field(default_factory=dict)
    session_data: dict = Field(default_factory=dict)
    document_context: dict = Field(default_factory=dict)
    concept_graph_info: dict = Field(default_factory=dict)


class ConceptResult(_Response):
    concept: str
    data: Any = None


class Health(_Response):
    status: str
    version: str
    api_version: str
    protocols: List[str]
    supported_types: List[str] = Field(default_factory=list)
    jobs_pending: int = 0


# ----------------------------------------------------------------------------- structure and knowledge
class SectionOut(_Response):
    id: str
    ordinal: int
    title: str
    depth: int
    parent_id: Optional[str] = None
    page_start: int
    page_end: int


class StructureOut(_Response):
    doc_id: str
    title: str
    format: str
    pages: int
    sections: List[SectionOut]


class ConceptOut(_Response):
    id: str
    name: str
    introduced_ordinal: int
    section_id: str = ""
    definition: str = ""
    importance: int = 0
    aliases: List[str] = Field(default_factory=list)


class RelationOut(_Response):
    src: str
    dst: str
    type: str
    strength: int = 1


class KnowledgeOut(_Response):
    as_of: Optional[int] = None
    concepts: List[ConceptOut]
    relations: List[RelationOut]


class SectionDependencyOut(_Response):
    section_id: str
    title: str
    concepts: List[str]
    weight: int


class SectionKnowledgeOut(_Response):
    section_id: str
    title: str
    introduces: List[ConceptOut]
    uses_earlier: List[ConceptOut]
    depends_on: List[SectionDependencyOut]


class GrowthPointOut(_Response):
    step: int
    section_id: str
    ordinal_end: int
    words: int
    new_concepts: int
    total_concepts: int
    total_relations: int
    novelty: float
    reuse: float


class GrowthOut(_Response):
    points: List[GrowthPointOut]


# ----------------------------------------------------------------------------- the learner
class BeliefOut(_Response):
    concept_id: str
    name: str
    p_known: float
    evidence: int


class LearnerOut(_Response):
    level: str
    prior: float
    position: Optional[int] = None
    concepts: List[BeliefOut]


class ReviewItemOut(_Response):
    concept_id: str
    name: str
    p_known: float
    retention: float               # p_known decayed by the time since the concept was last practised (0..1)
    last_seen: float               # unix time of the last evidence
    due: bool                      # retention fell below the review threshold


class ReviewOut(_Response):
    items: List[ReviewItemOut]     # weakest retention first


class HistoryPointOut(_Response):
    at: float
    kind: str
    concept_id: str
    name: str
    before: float
    after: float


class HistoryOut(_Response):
    points: List[HistoryPointOut]  # oldest first


class ExportOut(_Response):
    filename: str
    markdown: str


class LimitParams(DocIdParams):
    limit: int = Field(default=50, ge=1, le=500)


class QuizQuestionOut(_Response):
    id: str
    concept_id: str
    concept: str
    question: str
    options: List[str]
    difficulty: float
    kind: str


class QuizOut(_Response):
    questions: List[QuizQuestionOut]


class DiagnosticOut(_Response):
    level: str
    prior: float
    questions: List[QuizQuestionOut]


class AnswerOut(_Response):
    correct: bool
    correct_index: int
    explanation: str = ""
    already_answered: bool = False


class Ack(_Response):
    ok: bool = True
