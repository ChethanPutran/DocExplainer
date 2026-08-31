from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from doc_explainer.core.document.builder.base import SimilarityResult


@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: dict
    
class VectorStore(Protocol):

    def add(
        self,
        documents: Sequence[VectorDocument]
    ) -> None:
        ...

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None
    ) -> list["SimilarityResult"]:
        ...