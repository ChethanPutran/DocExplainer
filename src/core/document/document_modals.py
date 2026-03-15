from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any
import os
import json

# ==========================================
# DATA MODEL (Dataclasses) with to_dict methods
# ==========================================

@dataclass
class Sentence:
    sen_id: int
    raw_text: str
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embeddings: Dict[str, list] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sen_id": self.sen_id,
            "raw_text": self.raw_text,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
            # Skip embeddings as they might be large numpy arrays
            # "embeddings": self.embeddings
        }

@dataclass
class Image:
    img_id: str
    raw_img: str
    caption: List[Sentence]
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    clip_embedding: list = field(default_factory=list)
    vision_embedding: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "img_id": self.img_id,
            "raw_img": self.raw_img,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
            # Skip embeddings as they might be large
            # "clip_embedding": self.clip_embedding,
            # "vision_embedding": self.vision_embedding
        }

@dataclass
class Paragraphs:
    para_id: int
    raw_text: str
    sentences: List[Sentence]
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "para_id": self.para_id,
            "raw_text": self.raw_text,
            "sentences": [s.to_dict() for s in self.sentences],
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
            # "embeddings": self.embeddings,
            # "index": self.index
        }

@dataclass
class Table:
    table_id: int
    data: str
    raw_text: str
    caption: List[Sentence]
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "table_id": self.table_id,
            "data": self.data,
            "raw_text": self.raw_text,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
        }

@dataclass
class Equation:
    equation_id: int
    raw_text: str
    caption: List[Sentence]
    start: int = 0
    end: int = 0
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "equation_id": self.equation_id,
            "raw_text": self.raw_text,
            "caption": [s.to_dict() for s in self.caption] if self.caption else [],
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else [0, 0, 0, 0],
        }

@dataclass
class Section:
    sec_id: int
    title: str
    raw_text: str
    page_start: int = 1
    paragraphs: List[Paragraphs] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    subsections: List[Section] = field(default_factory=list)
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sec_id": self.sec_id,
            "title": self.title,
            "raw_text": self.raw_text,
            "page_start": self.page_start,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "images": [img.to_dict() for img in self.images],
            "tables": [t.to_dict() for t in self.tables],
            "equations": [eq.to_dict() for eq in self.equations],
            "subsections": [sub.to_dict() for sub in self.subsections],
        }

@dataclass
class Document:
    doc_id: int
    path: str
    raw_text: str
    title: str
    sections: List[Section]
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)

    def get_title(self) -> str:
        return os.path.basename(self.path) if self.path else "Untitled"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "doc_id": self.doc_id,
            "path": self.path,
            "raw_text": self.raw_text,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            # Skip embeddings for now
            # "embeddings": self.embeddings,
            # "index": self.index
        }