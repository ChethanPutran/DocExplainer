from abc import ABC, abstractmethod
from typing import Optional
from ..models.structure import Document
from ..models.tree import DocumentTree


class DocumentBuilder(ABC):
    """Base interface for document builders"""
    
    @abstractmethod
    def build_tree(self, document: Document, target_section: Optional[str] = None) -> DocumentTree:
        """Build document tree"""
        pass
    
    @abstractmethod
    def create_vector_db(self, tree: DocumentTree, persist_directory: Optional[str] = None):
        """Create vector database from tree"""
        pass