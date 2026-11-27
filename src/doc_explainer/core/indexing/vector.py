"""ChromaDB behind the VectorIndex port: one collection per document, cosine similarity.

Each chunk carries where it sits (reading-order ordinal, page, section), so a search can be limited
to ``ordinal <= position``: what the reader has already read.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Sequence

import chromadb
from chromadb.config import Settings

from ..common.ports import Chunk, Hit

_BATCH = 500


def _collection_name(doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)[:40].strip("_-") or "doc"
    return f"d_{safe}_{hashlib.sha1(doc_id.encode()).hexdigest()[:10]}"   # unique even if two ids sanitise alike


class ChromaVectorIndex:
    def __init__(self, path: Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))

    def _collection(self, doc_id: str, create: bool = True):
        name = _collection_name(doc_id)
        if create:
            return self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        try:
            return self._client.get_collection(name)
        except Exception:
            return None

    def add(self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        col = self._collection(doc_id)
        for i in range(0, len(chunks), _BATCH):
            part = chunks[i:i + _BATCH]
            col.upsert(
                ids=[c.id for c in part], documents=[c.text for c in part],
                embeddings=[list(map(float, v)) for v in vectors[i:i + _BATCH]],
                metadatas=[{"node_id": c.node_id, "kind": c.kind, "ordinal": c.ordinal, "page": c.page,
                            "section_id": c.section_id, "section_path": c.section_path} for c in part])

    def search(self, doc_id: str, vector: Sequence[float], k: int = 5, max_ordinal: Optional[int] = None) -> List[Hit]:
        col = self._collection(doc_id, create=False)
        if col is None or col.count() == 0:
            return []
        where = {"ordinal": {"$lte": int(max_ordinal)}} if max_ordinal is not None else None
        res = col.query(query_embeddings=[list(map(float, vector))], n_results=min(k, col.count()), where=where)
        hits = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=cid, text=text, node_id=meta["node_id"], kind=meta["kind"], ordinal=meta["ordinal"],
                          page=meta["page"], section_id=meta.get("section_id", ""), section_path=meta.get("section_path", ""))
            hits.append(Hit(chunk, max(0.0, min(1.0, 1.0 - float(dist)))))
        return hits

    def count(self, doc_id: str) -> int:
        col = self._collection(doc_id, create=False)
        return col.count() if col is not None else 0

    def delete(self, doc_id: str) -> None:
        try:
            self._client.delete_collection(_collection_name(doc_id))
        except Exception:
            pass
