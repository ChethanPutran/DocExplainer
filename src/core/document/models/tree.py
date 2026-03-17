from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import numpy as np
from .metadata import Metadata
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptRelationship


class ChunkType(Enum):
    """Type of document chunk"""
    DOCUMENT = auto()
    SECTION = auto()
    PARAGRAPH = auto()
    SENTENCE = auto()


class ChunkLevel(Enum):
    """Hierarchical level of chunk"""
    DOCUMENT = 0
    SECTION = 1
    PARAGRAPH = 2
    SENTENCE = 3


@dataclass
class DocumentChunk:
    """A chunk of document content"""
    text: str
    chunk_type: ChunkType = ChunkType.DOCUMENT
    level: ChunkLevel = ChunkLevel.DOCUMENT
    chunk_id: str = ""
    summary: str = ""
    parent_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Metadata] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            import uuid
            self.chunk_id = str(uuid.uuid4())[:8]


class DocumentNode:
    """Node in document tree hierarchy"""
    
    def __init__(self, node_id: str, chunk: DocumentChunk):
        self.id = node_id
        self.chunk = chunk
        self.children: Dict[str, 'DocumentNode'] = {}
        self.concepts: List[Concept] = []
        self.concept_relationships: List[ConceptRelationship] = []
    
    def add_child(self, child: 'DocumentNode'):
        """Add a child node"""
        self.children[child.id] = child
    
    def get_child(self, child_id: str) -> Optional['DocumentNode']:
        """Get child by ID"""
        return self.children.get(child_id)
    
    def get_all_children(self) -> List['DocumentNode']:
        """Get all children"""
        return list(self.children.values())


class DocumentTree:
    """Tree representation of document hierarchy"""
    
    def __init__(self, title: str, root: DocumentNode):
        self.title = title
        self.root = root
        self.hierarchy: Dict[str, List[DocumentChunk]] = {
            "document": [],
            "sections": [],
            "paragraphs": [],
            "sentences": []
        }
        self.summaries: List[str] = []
        self.total_chunks = 0

    def set_chunks(self, document_chunk: DocumentChunk,
                   section_chunks: List[DocumentChunk],
                   paragraph_chunks: List[DocumentChunk],
                   sentence_chunks: List[DocumentChunk]):
        """Set chunks in hierarchy"""
        self.hierarchy = {
            "document": [document_chunk],
            "sections": section_chunks,
            "paragraphs": paragraph_chunks,
            "sentences": sentence_chunks,
        }
        self.total_chunks = 1 + len(section_chunks) + len(paragraph_chunks) + len(sentence_chunks)

    def get_hierarchy(self) -> Dict[str, List[DocumentChunk]]:
        """Get hierarchy dictionary"""
        return self.hierarchy

    def get_sections(self) -> List[DocumentNode]:
        """Get all section nodes"""
        return list(self.root.children.values())

    def get_section(self, section_id: str) -> Optional[DocumentNode]:
        """Get section by ID"""
        return self.root.children.get(section_id)

    def get_previous_sections(self, section_id: str) -> List[DocumentNode]:
        """Get sections before given section"""
        sections = self.get_sections()
        result = []
        for section in sections:
            if section.id == section_id:
                break
            result.append(section)
        return result

    def get_previous_sections_summaries(self, section_id: str) -> List[str]:
        """Get summaries of sections before given section"""
        summaries = []
        for section in self.get_previous_sections(section_id):
            if section.chunk.summary:
                summaries.append(section.chunk.summary)
        return summaries

    def get_title(self) -> str:
        """Get document title"""
        return self.title

    def get_total_chunks(self) -> int:
        """Get total number of chunks"""
        return self.total_chunks


def create_empty_tree(title: str) -> DocumentTree:
    """Create an empty document tree"""
    from .metadata import SimpleMetadataCreator
    
    metadata_creator = SimpleMetadataCreator()
    chunk = DocumentChunk(
        text=title,
        chunk_type=ChunkType.DOCUMENT,
        level=ChunkLevel.DOCUMENT,
        metadata=metadata_creator.create_metadata(length=len(title))
    )
    root = DocumentNode("root", chunk)
    return DocumentTree(title, root)