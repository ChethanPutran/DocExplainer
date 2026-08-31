from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import numpy as np

from ...knowledge.models.relationship import ConceptNode
from .metadata import Metadata


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...knowledge import Concept, ConceptRelationship


class ChunkType(Enum):
    """Type of document chunk"""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


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
    summary: str = ""
    chunk_type: ChunkType = ChunkType.DOCUMENT
    level: ChunkLevel = ChunkLevel.DOCUMENT
    chunk_id: str = ""
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
        self.hierarchy: Dict[str, List[DocumentChunk]] = field(default_factory=dict)
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

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DocumentTree:
        """Create DocumentTree from dictionary"""

        def _parse_chunk_type(value: Any) -> ChunkType:
            if isinstance(value, ChunkType):
                return value
            if isinstance(value, str):
                normalized = value.upper()
                if hasattr(ChunkType, normalized):
                    return ChunkType[normalized]
            return ChunkType.DOCUMENT

        def _parse_chunk_level(value: Any) -> ChunkLevel:
            if isinstance(value, ChunkLevel):
                return value
            if isinstance(value, str):
                normalized = value.upper()
                if hasattr(ChunkLevel, normalized):
                    return ChunkLevel[normalized]
            if isinstance(value, int):
                for level in ChunkLevel:
                    if level.value == value:
                        return level
            return ChunkLevel.DOCUMENT

        def _deserialize_chunk(chunk_data: Any) -> DocumentChunk:
            if isinstance(chunk_data, DocumentChunk):
                return chunk_data
            if not isinstance(chunk_data, dict):
                return DocumentChunk(text="")

            embedding = chunk_data.get("embedding")
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            return DocumentChunk(
                text=str(chunk_data.get("text", "")),
                chunk_type=_parse_chunk_type(chunk_data.get("chunk_type", ChunkType.DOCUMENT)),
                level=_parse_chunk_level(chunk_data.get("level", ChunkLevel.DOCUMENT)),
                chunk_id=str(chunk_data.get("chunk_id", "")),
                summary=str(chunk_data.get("summary", "")),
                parent_id=chunk_data.get("parent_id"),
                embedding=embedding,
                metadata=chunk_data.get("metadata"),
            )

        def _deserialize_node(node_data: Any, parent: Optional[DocumentNode] = None) -> DocumentNode:
            if isinstance(node_data, DocumentNode):
                return node_data
            if not isinstance(node_data, dict):
                node_data = {}

            node_id = str(node_data.get("id") or node_data.get("node_id") or "root")
            if not node_id and parent is not None:
                node_id = parent.id

            chunk_data = node_data.get("chunk") or {}
            if not isinstance(chunk_data, dict):
                chunk_data = {}

            chunk = _deserialize_chunk(chunk_data)
            if not chunk.chunk_id:
                chunk.chunk_id = node_id
            if parent is not None and chunk.parent_id is None:
                chunk.parent_id = parent.id

            node = DocumentNode(node_id, chunk)
            node.concepts = list(node_data.get("concepts", []))
            node.concept_relationships = list(node_data.get("concept_relationships", []))

            children_data = node_data.get("children", {})
            if isinstance(children_data, list):
                for child_data in children_data:
                    child = _deserialize_node(child_data, node)
                    node.add_child(child)
            elif isinstance(children_data, dict):
                for child_data in children_data.values():
                    child = _deserialize_node(child_data, node)
                    node.add_child(child)

            return node

        if not isinstance(data, dict):
            data = {}

        title = str(data.get("title", ""))
        root_data = data.get("root")
        root_node = _deserialize_node(root_data)
        tree = DocumentTree(title, root_node)

        hierarchy_data = data.get("hierarchy")
        if isinstance(hierarchy_data, dict):
            tree.hierarchy = {
                "document": [_deserialize_chunk(chunk_data) for chunk_data in hierarchy_data.get("document", [])],
                "sections": [_deserialize_chunk(chunk_data) for chunk_data in hierarchy_data.get("sections", [])],
                "paragraphs": [_deserialize_chunk(chunk_data) for chunk_data in hierarchy_data.get("paragraphs", [])],
                "sentences": [_deserialize_chunk(chunk_data) for chunk_data in hierarchy_data.get("sentences", [])],
            }

        tree.summaries = list(data.get("summaries", []))
        tree.total_chunks = int(data.get("total_chunks", tree.total_chunks))

        return tree
    
