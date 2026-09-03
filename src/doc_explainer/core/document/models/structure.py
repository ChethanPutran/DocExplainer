from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Generator
from .base import Serializable, Identifiable
from .content import Paragraph, Image, Table, Equation
import uuid

@dataclass
class FontInfo:
    """Font information for text span"""
    size: float
    name: str
    flags: int
    
    @property
    def is_bold(self) -> bool:
        return ("Bold" in self.name) or bool(self.flags & 16)
    
@dataclass
class Section(Serializable, Identifiable):
    """Represents a section in a document"""
    section_id: str
    document_id: str
    level: int
    page: int
    start: int = 0
    end: int = 0
    title: str = ""
    text: str = ""
    paragraphs: List[Paragraph] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.section_id
    
    def add_paragraph(self, paragraph: Paragraph):
        """Add a paragraph to the section"""
        self.paragraphs.append(paragraph)
        self.text += paragraph.text + "\n"
    
    def add_subsection(self, section: 'Section'):
        """Add a subsection"""
        self.subsections.append(section)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "level": self.level,
            "page": self.page,
            "start": self.start,
            "end": self.end,
            "title": self.title,
            "text": self.text,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Section':
        return cls(
            section_id=data.get("id", ""),
            document_id=data.get("document_id", ""),
            level=data.get("level", 0),
            page=data.get("page", 0),
            start=data.get("start", 0),
            end=data.get("end", 0),
            title=data.get("title", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )

@dataclass
class Document(Serializable, Identifiable):
    """Represents a complete document"""
    path: Path
    title: str
    text: str = ""
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return str(self.metadata.get("document_id", self.path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.id,
            "path": str(self.path),
            "title": self.title,
            "text": self.text,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        metadata = dict(data.get("metadata", {}))
        metadata.setdefault("document_id", data.get("document_id"))
        return cls(
            path=Path(data["path"]),
            title=data.get("title", ""),
            text=data.get("text", ""),
            metadata=metadata,
        )
    
    def get_all_paragraphs(self) -> Generator[Paragraph, None, None]:
        for sec in self.sections:
            yield from sec.paragraphs

    def get_text_generator(self) -> Generator[str, None, None]:
        for sec in self.sections:
            yield sec.text