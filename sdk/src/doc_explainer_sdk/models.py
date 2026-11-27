"""Typed results. These mirror the server's wire format; the SDK owns its copy so it never has to
import the backend. Unknown fields from a newer server are ignored."""
from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ExplanationLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExplanationDepth(str, Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"


class ExplanationStyle(_Model):
    """How to explain: pass one to ``explain(style=...)``; results carry the style that was used."""
    level: str
    depth: str
    description: str = ""
    examples: Optional[List[Any]] = None


class ResourceSuggestion(_Model):
    concept: str
    resource_type: str
    difficulty: str


class ExplanationMetadata(_Model):
    estimated_complexity: float = 0.0
    user_knowledge_matched: bool = False
    gap_bridging: bool = False
    processing_time_ms: float = 0.0


class Resource(_Model):
    title: str
    url: str
    type: str
    description: str = ""
    difficulty: Any = None
    source: str = ""            # "This document", "YouTube", "Wikipedia", ...


class Explanation(_Model):
    explanation: str
    style: Optional[ExplanationStyle] = None
    context_used: dict = Field(default_factory=dict)
    known_concepts_used: List[str] = Field(default_factory=list)
    unknown_concepts_explained: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    suggested_resources: List[ResourceSuggestion] = Field(default_factory=list)
    metadata: Optional[ExplanationMetadata] = None
    resources: List[Resource] = Field(default_factory=list)


class ExplanationResult(_Model):
    text: str
    explanation: Explanation
    processing_time_ms: float = 0.0


class Document(_Model):
    doc_id: str
    title: str
    filename: str
    size_bytes: int
    uploaded_at: str
    knowledge: str = "none"     # ready | partial | failed | none: was the knowledge graph built?
    sections: int = 0
    concepts: int = 0


class JobError(_Model):
    code: str
    message: str


class Job(_Model):
    """A document registration running on the server.

    ``state``: queued -> running -> succeeded | failed | cancelled; paused is a rest stop that ``resume_job`` continues from. When succeeded, ``document`` is set; when
    failed, ``error`` is. ``reused`` means identical content was already registered (nothing ran).
    """
    job_id: str
    state: str
    filename: str
    created_at: str
    kind: str = "register_document"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    queue_position: Optional[int] = None  # 1 = next to run, while queued
    reused: bool = False
    progress: float = 0.0       # 0..1 across parsing, indexing and building the knowledge graph
    stage: str = ""             # what the server is doing now
    document: Optional[Document] = None
    error: Optional[JobError] = None

    @property
    def done(self) -> bool:
        return self.state in ("succeeded", "failed", "cancelled")


class Context(_Model):
    user_knowledge: dict = Field(default_factory=dict)
    session_data: dict = Field(default_factory=dict)
    document_context: dict = Field(default_factory=dict)
    concept_graph_info: dict = Field(default_factory=dict)


class ConceptResult(_Model):
    concept: str
    data: Any = None


class Health(_Model):
    status: str
    version: str
    api_version: str
    protocols: List[str] = Field(default_factory=list)
    supported_types: List[str] = Field(default_factory=list)  # file extensions the server can read
    jobs_pending: int = 0


# ----------------------------------------------------------------------------- the document's structure and knowledge
class Section(_Model):
    id: str
    ordinal: int                # reading-order position; this is what "section_id" means everywhere
    title: str
    depth: int
    parent_id: Optional[str] = None
    page_start: int
    page_end: int


class Structure(_Model):
    doc_id: str
    title: str
    format: str
    pages: int
    sections: List[Section]


class ConceptInfo(_Model):
    id: str
    name: str
    introduced_ordinal: int
    section_id: str = ""
    definition: str = ""
    importance: int = 0
    aliases: List[str] = Field(default_factory=list)


class Relation(_Model):
    src: str
    dst: str
    type: str                   # prerequisite_of | part_of | example_of | contrasts_with | related_to
    strength: int = 1


class Knowledge(_Model):
    """The knowledge graph as a reader at ``as_of`` would know it (None = the whole document)."""
    as_of: Optional[int] = None
    concepts: List[ConceptInfo]
    relations: List[Relation]


class SectionDependency(_Model):
    section_id: str
    title: str
    concepts: List[str]
    weight: int


class SectionKnowledge(_Model):
    section_id: str
    title: str
    introduces: List[ConceptInfo]
    uses_earlier: List[ConceptInfo]
    depends_on: List[SectionDependency]     # earlier sections this one relies on


class GrowthPoint(_Model):
    step: int
    section_id: str
    ordinal_end: int
    words: int
    new_concepts: int
    total_concepts: int
    total_relations: int
    novelty: float
    reuse: float


# ----------------------------------------------------------------------------- the learner
class Belief(_Model):
    concept_id: str
    name: str
    p_known: float
    evidence: int


class Learner(_Model):
    level: str
    prior: float
    position: Optional[int] = None
    concepts: List[Belief]


class ReviewItem(_Model):
    concept_id: str
    name: str
    p_known: float
    retention: float
    last_seen: float
    due: bool


class HistoryPoint(_Model):
    at: float
    kind: str
    concept_id: str
    name: str = ""
    before: float = 0.0
    after: float = 0.0


class Export(_Model):
    filename: str
    markdown: str


class QuizQuestion(_Model):
    id: str
    concept_id: str
    concept: str
    question: str
    options: List[str]
    difficulty: float
    kind: str                   # practice | diagnostic


class Diagnostic(_Model):
    level: str
    prior: float
    questions: List[QuizQuestion]


class AnswerResult(_Model):
    correct: bool
    correct_index: int
    explanation: str = ""
    already_answered: bool = False


# ----------------------------------------------------------------------------- conversations
class ChatResult(ExplanationResult):
    session_id: str = ""
    message_id: int = 0


class Session(_Model):
    id: str
    doc_id: str
    title: str
    created_at: float
    updated_at: float
    messages: int = 0
    notes: int = 0


class Message(_Model):
    id: int
    role: str                   # user | assistant
    text: str
    kind: str                   # ask | explain | summarize
    created_at: float
    meta: dict = Field(default_factory=dict)       # follow_ups, section_id, concept_ids, resources


class Note(_Model):
    id: str
    kind: str                   # note | summary
    text: str
    created_at: float


class SessionDetail(_Model):
    session: Session
    messages: List[Message]
    notes: List[Note]


class StreamEvent(_Model):
    """One event of a streamed reply. ``type``: "session" (data: the Session), "status" (data: stage, text),
    "delta" (data: text), "done" (data: the ChatResult)."""
    type: str
    data: dict = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.data.get("text", ""))

    @property
    def result(self) -> Optional[ChatResult]:
        return ChatResult.model_validate(self.data) if self.type == "done" else None
