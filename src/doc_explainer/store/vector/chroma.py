from typing import Optional, Sequence

from base import VectorDocument, VectorStore

class ChromaVectorStore(VectorStore):

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
    ):
        ...