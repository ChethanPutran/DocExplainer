from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from ...core.document.models.structure import Document
from ...core.document.models.tree import DocumentTree


class BaseDocumentRepository(ABC):
    """Base interface for document storage"""

    @abstractmethod
    def save_document(self, file_path: Path, doc_id: str) -> bool:
        """Save document"""
        pass

    @abstractmethod
    def save_document_model(self, document: Document, doc_id: str) -> bool:
        """Save a parsed document model."""
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass

    @abstractmethod
    def save_document_tree(self, tree: DocumentTree, doc_id: str) -> bool:
        """Save document tree"""
        pass

    @abstractmethod
    def get_document_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Get document tree by ID"""
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        pass

    @abstractmethod
    def list_documents(self) -> list:
        """List all document IDs"""
        pass


class BaseDocumentCache(ABC):
    """In-memory cache for documents"""
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        """Set item in cache"""
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        pass

    @abstractmethod
    def clear(self):
        """Clear cache"""
        pass
