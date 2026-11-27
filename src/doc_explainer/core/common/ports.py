"""Interfaces the core packages use to talk to the outside world. The orchestrator supplies the
implementations, so no core package imports another, or any concrete store or model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Raises on failure: there is deliberately no silent fallback."""
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


@runtime_checkable
class JsonLLM(Protocol):
    """An LLM that answers with a JSON object. Rate limiting and retries live behind it."""

    def ask_json(self, system: str, user: str) -> Dict[str, Any]: ...


@dataclass
class Chunk:
    """A piece of text to index, with where it sits in the document."""
    id: str
    text: str
    node_id: str
    kind: str
    ordinal: int            # reading order of the chunk's first node
    page: int
    section_id: str = ""
    section_path: str = ""


@dataclass
class Hit:
    chunk: Chunk
    score: float            # similarity in [0, 1]; higher is closer


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, doc_id: str, vector: Sequence[float], k: int = 5,
               max_ordinal: Optional[int] = None) -> List[Hit]: ...

    def count(self, doc_id: str) -> int: ...

    def delete(self, doc_id: str) -> None: ...


class LLMError(RuntimeError):
    """The model call failed (network, auth, quota, ...). Not worth retrying with a different prompt."""


class LLMFormatError(LLMError):
    """The model answered, but not with the JSON we asked for (empty or cut off), even after retries with a bigger budget."""



@runtime_checkable
class TextLLM(Protocol):
    """A model whose plain-text reply can be read as it is written."""

    def stream_text(self, system: str, user: str) -> Iterator[str]: ...
