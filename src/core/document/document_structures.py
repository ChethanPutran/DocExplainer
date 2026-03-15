from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import uuid
import numpy as np


@dataclass
class MetaData:
    length: int = 0
    start: int = 0
    end: int = 0
    is_concept: bool = False
    page: int = 0 


class ChunkType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class ChunkLevel(Enum):
    DOCUMENT = 0
    SECTION = 1
    PARAGRAPH = 2
    SENTENCE = 3


@dataclass
class DocumentChunk:
    text: str
    chunk_type: ChunkType = ChunkType.DOCUMENT
    level: ChunkLevel = ChunkLevel.DOCUMENT
    chunk_id: int  = 0
    summary: str = ''
    parent_id: int | None = None
    embedding: np.ndarray | None = None
    metadata: MetaData | None = None


class MetaDataCreator:
    def create_metadata(
        self, length: int, start: int, end: int, is_concept: bool = False
    ) -> MetaData:
        raise NotImplementedError


class SimpleMetaDataCreator(MetaDataCreator):
    def create_metadata(
        self, length: int, start: int, end: int, is_concept: bool = False, page:int = 0
    ) -> MetaData:
        return MetaData(
            length=length,
            start=start,
            end=end,
            is_concept=is_concept,
            page=page
        )


class DocumentNode:
    def __init__(self,id,chunk: DocumentChunk):
        self.id = id
        self.chunk = chunk
        self.children: Dict[int,DocumentNode] = {}
        self.concepts = []
        self.concept_relationships = []


class DocumentTree:
    def __init__(self, title: str, root: DocumentNode):
        self.title = title
        self.root = root
        self.hierarchy: Dict = {}
        self.summaries: List[str] = []
        self.total_chunks = 0

    def set_chunks(
        self,
        document_chunk: DocumentChunk,
        section_chunks: List[DocumentChunk],
        paragraph_chunks: List[DocumentChunk],
        sentence_chunks: List[DocumentChunk],
    ):
        self.hierarchy = {
            "document": document_chunk,
            "sections": section_chunks,
            "paragraphs": paragraph_chunks,
            "sentences": sentence_chunks,
        }
        self.total_chunks = 1 + len(section_chunks) + len(paragraph_chunks) + len(sentence_chunks)

    def get_hierarchy(self) -> Dict:
        return self.hierarchy

    def get_sections_chunks(self) -> List[DocumentChunk]:
        return self.hierarchy.get("sections", [])

    def get_section(self, section_id: int) -> DocumentNode:
        return self.root.children[section_id]

    def get_previous_sections_summaries(self,section_id)->List[str]:
        return self.summaries[:section_id]

    def get_previous_sections(self, section_id: int) -> List[DocumentChunk]:
        return self.get_sections_chunks()[:section_id]
    
    def get_sections(self) -> List[DocumentNode]:
        return list(self.root.children.values())

    def get_title(self) -> str:
        return self.title

    def get_total_chunks(self) -> int:
        return self.total_chunks
