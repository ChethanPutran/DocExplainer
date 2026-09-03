from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from .base import Serializable, Positionable
import uuid


@dataclass
class Sentence(Serializable, Positionable):
    """Represents a sentence in a document"""
    id: str 
    text: str
    start: int
    end: int
    page: int
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sentence':
        values = dict(data)
        values["bbox"] = tuple(values.get("bbox", (0, 0, 0, 0)))
        return cls(**values)
    

@dataclass
class Paragraph(Serializable, Positionable):
    """Represents a paragraph in a document"""
    id: str
    text: str
    start: int
    end: int
    page: int
    bbox: Tuple[float, float, float, float]
    sentences: List[Sentence] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox),
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paragraph':
        values = dict(data)
        values["bbox"] = tuple(values.get("bbox", (0, 0, 0, 0)))
        values["sentences"] = [Sentence.from_dict(item) for item in values.get("sentences", [])]
        return cls(**values)
    
    def add_sentence(self, sentence: Sentence):
        """Add a sentence to the paragraph"""
        self.sentences.append(sentence)
        if not self.start:
            self.start = sentence.start
        self.end = sentence.end


@dataclass
class Image(Serializable, Positionable):
    """Represents an image in a document"""
    image_path: str
    caption: Optional[List[Sentence]] = None
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    image_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Image':
        data["caption"] = [Sentence.from_dict(s) for s in data.get("caption", [])]
        if "bbox" in data and isinstance(data["bbox"], list):
            data["bbox"] = tuple(data["bbox"])
        return cls(**data)
    
    @property
    def id(self) -> str:
        return self.image_id
    
    @property
    def start(self) -> int:
        return 0  # Images don't have text position
    
    @property
    def end(self) -> int:
        return 0  # Images don't have text position


@dataclass
class Table(Serializable, Positionable):
    """Represents a table in a document"""
    data: str
    text: str
    caption: Optional[List[Sentence]] = None
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    table_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "data": self.data,
            "text": self.text,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Table':
        data["caption"] = [Sentence.from_dict(s) for s in data.get("caption", [])]
        if "bbox" in data and isinstance(data["bbox"], list):
            data["bbox"] = tuple(data["bbox"])
        return cls(**data)
    
    @property
    def id(self) -> str:
        return self.table_id


@dataclass
class Equation(Serializable, Positionable):
    """Represents an equation in a document"""
    text: str
    caption: Optional[List[Sentence]] = None
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    equation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "text": self.text,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Equation':
        data["caption"] = [Sentence.from_dict(s) for s in data.get("caption", [])]
        if "bbox" in data and isinstance(data["bbox"], list):
            data["bbox"] = tuple(data["bbox"])
        return cls(**data)
    
    @property
    def id(self) -> str:
        return self.equation_id