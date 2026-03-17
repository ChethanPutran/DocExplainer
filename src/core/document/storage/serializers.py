from typing import Dict, Any
from ..models.structure import Document
from ..models.content import Sentence, Paragraph, Image, Table, Equation


class DocumentSerializer:
    """Serializer for Document objects"""
    
    @staticmethod
    def serialize(document: Document) -> Dict[str, Any]:
        """Serialize document to dictionary"""
        return document.to_dict()
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Document:
        """Deserialize document from dictionary"""
        return Document.from_dict(data)


class TreeSerializer:
    """Serializer for DocumentTree objects"""
    
    @staticmethod
    def serialize(tree) -> Dict[str, Any]:
        """Serialize tree to dictionary"""
        return {
            "title": tree.title,
            "total_chunks": tree.total_chunks,
            "hierarchy": {
                level: [chunk.chunk_id for chunk in chunks]
                for level, chunks in tree.hierarchy.items()
            }
        }