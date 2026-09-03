from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from ..models.structure import Document, Section
from ..models.base import DocumentMetadata


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

    @abstractmethod
    def parse_metadata(
            self,
            file_path: Path
        ) -> DocumentMetadata:
        """Parse only metadata from document"""
        pass

    @abstractmethod
    def iter_sections(
        self,
        file_path: Path
    ) -> Iterator[Section]:
        """Iterate over sections in the document"""
        pass