"""Indexer: one DocumentStructure in, three derived stores out.

    text + structure  -> JsonContentStore      (source of truth)
    structure only    -> SqliteStructureStore  (graph store, no body text)
    chunks            -> embeddings -> VectorIndex

Re-indexing a document replaces what was there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..common.ports import Embedder, VectorIndex
from ..common.structure import DocumentStructure
from .chunker import chunk_structure, embedding_text
from .stores import JsonContentStore, SqliteStructureStore

_BATCH = 64


@dataclass
class IndexReport:
    nodes: int
    sections: int
    chunks: int


class Indexer:
    def __init__(self, embedder: Embedder, vectors: VectorIndex, structure_store: SqliteStructureStore,
                 content_store: JsonContentStore):
        self.embedder, self.vectors, self.structure_store, self.content_store = embedder, vectors, structure_store, content_store

    def index(self, structure: DocumentStructure, progress: Optional[Callable[[str, float], None]] = None) -> IndexReport:
        say = progress or (lambda stage, fraction: None)
        say("storing structure", 0.0)
        self.content_store.save(structure)
        self.structure_store.save(structure.without_text())
        chunks = chunk_structure(structure)
        self.vectors.delete(structure.doc_id)
        for i in range(0, len(chunks), _BATCH):
            part = chunks[i:i + _BATCH]
            self.vectors.add(structure.doc_id, part, self.embedder.embed([embedding_text(c) for c in part]))
            say("embedding", min(1.0, (i + len(part)) / max(len(chunks), 1)))
        say("indexed", 1.0)
        return IndexReport(nodes=len(structure.nodes), sections=len(structure.sections()), chunks=len(chunks))

    def remove(self, doc_id: str) -> None:
        self.vectors.delete(doc_id)
        self.structure_store.delete(doc_id)
        self.content_store.delete(doc_id)
