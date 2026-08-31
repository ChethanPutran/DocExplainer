from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from ..models.structure import Document
from ..models.tree import DocumentTree
from typing import Optional, List, Protocol, runtime_checkable


    
class BaseDocumentEngine(ABC):
    """Base interface for document engine"""

    @abstractmethod
    def ingest_and_map(self, document: Document, target_query: Optional[str] = None) -> DocumentTree:
        """
        Full processing pipeline:
        1. Full indexing
        2. Target discovery
        3. Tree building
        4. Tree indexing
        """
        pass

    @abstractmethod
    def query(self, user_query: str, level: str = "paragraph", k: int = 3) -> list:
        """Search within hierarchical summaries"""
        pass

    @abstractmethod
    def get_document_tree(self) -> Optional[DocumentTree]:
        """Get the built document tree"""
        pass

@runtime_checkable
class SimilarityResult(Protocol):
    page_content: str

@runtime_checkable
class SimilaritySearchDB(Protocol):
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None,
    ) -> List[SimilarityResult]:
        ...

