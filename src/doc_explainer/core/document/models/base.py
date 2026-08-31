from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid


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


class Identifiable(ABC):
    """Base interface for identifiable objects"""
    
    @property
    @abstractmethod
    def id(self) -> Any:
        """Get object ID"""
        pass


class Positionable(ABC):
    """Base interface for objects with position"""
    
    @property
    @abstractmethod
    def start(self) -> int:
        """Get start position"""
        pass
    
    @property
    @abstractmethod
    def end(self) -> int:
        """Get end position"""
        pass
    
    @property
    @abstractmethod
    def page(self) -> int:
        """Get page number"""
        pass
    
    @property
    @abstractmethod
    def bbox(self) -> tuple:
        """Get bounding box"""
        pass


@dataclass
class ProcessingContext:
    document_id: str
    previous_section_summary: Optional[str] = None
    section_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInfo:
    document_id: str
    source_path: str = ""
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None

    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())

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
    parent_section_id: Optional[str]
    paragraphs: list["ProcessedParagraph"] = field(
        default_factory=list
    )
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