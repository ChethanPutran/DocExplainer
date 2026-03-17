from abc import ABC, abstractmethod
from typing import Optional, Any
from ..models.structure import Document
from ..models.tree import DocumentTree


class DocumentStorage(ABC):
    """Base interface for document storage"""
    
    @abstractmethod
    def save_document(self, document: Document, doc_id: str) -> bool:
        """Save document"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    def save_tree(self, tree: DocumentTree, doc_id: str) -> bool:
        """Save document tree"""
        pass
    
    @abstractmethod
    def get_tree(self, doc_id: str) -> Optional[DocumentTree]:
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