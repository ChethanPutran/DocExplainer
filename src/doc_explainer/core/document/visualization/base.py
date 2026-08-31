from abc import ABC, abstractmethod
from ..models.structure import Document
from ..models.tree import DocumentTree


class DocumentVisualizer(ABC):
    """Base interface for document visualizers"""
    
    @abstractmethod
    def visualize_document(self, document: Document):
        """Visualize document"""
        pass
    
    @abstractmethod
    def visualize_tree(self, tree: DocumentTree):
        """Visualize document tree"""
        pass