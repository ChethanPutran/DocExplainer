from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

# ==========================================
# DATA MODEL (Dataclasses)
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

@dataclass
class Image:
    img_id: int
    raw_img: str  # Filename or path
    caption: str
    page: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embeddings: Dict[str, list] = field(default_factory=dict)

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

@dataclass
class Section:
    sec_id: int
    title: str
    raw_text: str
    page_start: int = 1
    paragraphs: List[Paragraphs] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    subsections: List[Section] = field(default_factory=list)
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)

@dataclass
class Document:
    doc_id: int
    path: str
    raw_text: str
    sections: List[Section]
    embeddings: Dict[str, list] = field(default_factory=dict)
    index: List[Dict] = field(default_factory=list)

    def get_title(self) -> str:
        return os.path.basename(self.path) if self.path else "Untitled"