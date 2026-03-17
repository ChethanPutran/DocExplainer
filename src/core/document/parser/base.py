from abc import ABC, abstractmethod
from typing import Optional
from ..models.structure import Document


class DocumentParser(ABC):
    """Base interface for document parsers"""
    
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """Parse document from file path"""
        pass
    
    @abstractmethod
    def to_json(self, document: Document, output_path: str):
        """Save document to JSON"""
        pass
    
    @abstractmethod
    def from_json(self, json_path: str) -> Optional[Document]:
        """Load document from JSON"""
        pass