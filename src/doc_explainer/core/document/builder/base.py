from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from doc_explainer.core.document.parser.base import DocumentParser
from doc_explainer.store.vector.base import SimilarityResult
from typing import Optional, List, Protocol, runtime_checkable


    
class BaseDocumentEngine(ABC):
    """Base interface for document engine"""
    parser: Optional[DocumentParser]

    @abstractmethod
    def ingest(self, file_path: Path, target_query: Optional[str] = None) -> str:
        """
        Full processing pipeline:
        1. Full indexing
        2. Target discovery
        3. Tree building
        4. Tree indexing
        """
        pass

    def use_parser(self, parser):
        """Set the parser to use for document ingestion"""
        self.parser = parser



@runtime_checkable
class SimilaritySearchDB(Protocol):
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[dict] = None,
    ) -> List[SimilarityResult]:
        ...


class TreeDBAdapter:
    """
    Adapter for similarity search databases to provide a consistent interface
    and support filtering if the underlying database does not support it.
    """
    def __init__(self, db):
        self._db = db

    def similarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None):
        # Prefer calling underlying method with filter if supported
        try:
            return self._db.similarity_search(query, k=k, filter=filter)
        except TypeError:
            # Underlying db doesn't accept filter kwarg: call basic search then apply filter
            results = self._db.similarity_search(query, k=k)
            if not filter:
                return results

            # Apply simple metadata filtering (supports equality checks)
            def matches(item):
                meta = getattr(item, 'metadata', {}) or {}
                for key, val in filter.items():
                    if meta.get(key) != val:
                        return False
                return True

            filtered = [r for r in results if matches(r)]
            return filtered[:k]

    # Provide passthrough for other attributes/methods
    def __getattr__(self, name):
        return getattr(self._db, name)
