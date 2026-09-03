from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Iterator, List, Optional, Protocol
from dataclasses import dataclass, field
import uuid

@dataclass
class SimilarityResult:
    document_id: str
    content: str
    score: float
    metadata: dict

@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: dict
    
class Serializable(ABC):
    """Base interface for serializable objects"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create from dictionary"""
        pass


class Identifiable(Protocol):
    """Base interface for identifiable objects"""
    id: Any


class Positionable(Protocol):
    """Base interface for objects with position"""
    start: int
    end: int
    page: int
    bbox: tuple

@dataclass(slots=True)
class Relationship:
    source_id: str
    target_id: str
    relation: str

    
@dataclass
class ProcessingContext:
    namespace: str
    previous_section_summary: Optional[str] = ''
    section_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class DocumentMetadata:
    document_id: str
    file_path: str
    filename: str
    title: str = ""
    author: str = ""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    subject: str = ""
    page_count: int = 0
    file_size: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Section:
    section_id: str
    document_id: str
    title: str

    text: str

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    parent_section_id: Optional[str] = None

    subsections: list["Section"] = field(default_factory=list)


@dataclass
class ProcessedSection:
    section_id: str
    document_id: str

    title: str
    summary: str

    vector_documents: Callable[[], Iterator[VectorDocument]]
    relationships: Callable[[], Iterator[Relationship]]
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedParagraph:
    paragraph_id: str
    text: str
    summary: str

    sentences: list["ProcessedSentence"] = field(
        default_factory=list
    )


@dataclass
class ProcessedSentence:
    sentence_id: str
    text: str