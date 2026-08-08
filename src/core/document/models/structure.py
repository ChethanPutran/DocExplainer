from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Generator
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
    title: str
    
    text: str = ""
    page_start: int = 1
    paragraphs: List[Paragraph] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    section_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "text": self.text,
            "page_start": self.page_start,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "images": [img.to_dict() for img in self.images],
            "tables": [t.to_dict() for t in self.tables],
            "equations": [eq.to_dict() for eq in self.equations],
            "subsections": [sub.to_dict() for sub in self.subsections],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Section':
        data["paragraphs"] = [Paragraph.from_dict(p) for p in data.get("paragraphs", [])]
        data["images"] = [Image.from_dict(i) for i in data.get("images", [])]
        data["tables"] = [Table.from_dict(t) for t in data.get("tables", [])]
        data["equations"] = [Equation.from_dict(e) for e in data.get("equations", [])]
        data["subsections"] = [cls.from_dict(sub) for sub in data.get("subsections", [])]
        return cls(**data)
    
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


@dataclass
class Document(Serializable, Identifiable):
    """Represents a complete document"""
    path: str
    title: str
    sections: List[Section]
    text: str = ""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "path": self.path,
            "title": self.title,
            "text": self.text,
            "sections": [s.to_dict() for s in self.sections],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        data["sections"] = [Section.from_dict(s) for s in data.get("sections", [])]
        return cls(**data)
    
    @property
    def id(self) -> str:
        return self.document_id
    
    def get_text_generator(self) -> Generator[str, None, None]:
        """Get a generator for all text in the document"""
        for section in self.sections:
            for paragraph in section.paragraphs:
                yield paragraph.text
    
    def get_all_paragraphs(self) -> List[Paragraph]:
        """Get all paragraphs in the document"""
        paragraphs = []
        for section in self.sections:
            paragraphs.extend(section.paragraphs)
            for subsection in section.subsections:
                paragraphs.extend(subsection.paragraphs)
        return paragraphs
    
    def get_all_images(self) -> List[Image]:
        """Get all images in the document"""
        images = []
        for section in self.sections:
            images.extend(section.images)
            for subsection in section.subsections:
                images.extend(subsection.images)
        return images