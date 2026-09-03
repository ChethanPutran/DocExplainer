from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

from doc_explainer.core.document.models.base import SimilarityResult, VectorDocument


class VectorStore(Protocol):

    def add(
        self,
        namespace: str,
        documents: Iterator[VectorDocument],
        batch_size: int = 32
    ) -> None:
        ...

    def search(
        self,
        namespace: str,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None
    ) -> list["SimilarityResult"]:
        ...