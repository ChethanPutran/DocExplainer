from __future__ import annotations

from typing import Iterator, Optional, Protocol

from doc_explainer.core.document.models.base import (
    ProcessedSection,
    Relationship,
)
from doc_explainer.core.document.models.tree import DocumentChunk


class GraphStore(Protocol):

    def add_document(
        self,
        document_id: str,
        title: str,
        namespace: str,
        metadata: Optional[dict] = None,
    ) -> None:
        ...

    def add_section(
        self,
        namespace: str,
        section: ProcessedSection,
    ) -> None:
        ...

    def add_chunk(
        self,
        namespace: str,
        chunk_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        ...

    def get_document(
        self,
        document_id: str,
    ) -> Optional[DocumentChunk]:
        ...

    def get_section(
        self,
        namespace: str,
        section_id: str,
    ) -> Optional[ProcessedSection]:
        ...

    def get_children(
        self,
        node_id: str,
    ) -> list[DocumentChunk]:
        ...

    def add_relationships(
        self,
        namespace: str,
        relationships: Iterator[Relationship],
        batch_size: int = 100,
    ) -> None:
        ...