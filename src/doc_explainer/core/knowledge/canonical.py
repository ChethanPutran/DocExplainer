"""Deciding that "Photosynthesis", "photosynthesis" and "the photosynthesis process" are one concept.

Cheap, deterministic checks first (normalised name and alias match); embedding similarity of names
only as a second step, with a high threshold, because merging two different concepts ("mitosis",
"meiosis") is worse than keeping two names for one.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence

from ..common.knowledge import Concept
from ..common.ports import Embedder

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_PAREN = re.compile(r"\s*[\(\[].*?[\)\]]")


def normalize_name(name: str) -> str:
    n = _PAREN.sub("", name).strip().lower()
    n = re.sub(r"[^\w\s\-]", " ", n)
    n = _ARTICLES.sub("", re.sub(r"\s+", " ", n).strip())
    words = n.split()
    if words:
        w = words[-1]
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("sses"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        words[-1] = w
    return " ".join(words)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Canonicalizer:
    def __init__(self, embedder: Embedder, threshold: float = 0.9):
        self.embedder, self.threshold = embedder, threshold
        self._by_norm: Dict[str, str] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._names: Dict[str, str] = {}

    def load(self, concepts: Iterable[Concept]) -> None:
        self._by_norm.clear(); self._vectors.clear(); self._names.clear()
        for c in concepts:
            self.add(c)

    def add(self, concept: Concept) -> None:
        self._names[concept.id] = concept.name
        for n in [concept.name, *concept.aliases]:
            self._by_norm.setdefault(normalize_name(n), concept.id)

    def add_alias(self, concept_id: str, alias: str) -> None:
        self._by_norm.setdefault(normalize_name(alias), concept_id)

    def match(self, names: Sequence[str]) -> Optional[str]:
        """The id of the existing concept any of ``names`` refers to, or None if it is new."""
        for n in names:
            hit = self._by_norm.get(normalize_name(n))
            if hit:
                return hit
        if not self._names:
            return None
        missing = [cid for cid in self._names if cid not in self._vectors]
        if missing:
            for cid, v in zip(missing, self.embedder.embed([self._names[c] for c in missing])):
                self._vectors[cid] = v
        probes = self.embedder.embed([names[0]])[0]
        best_id, best = None, 0.0
        for cid, v in self._vectors.items():
            s = _cosine(probes, v)
            if s > best:
                best_id, best = cid, s
        return best_id if best >= self.threshold else None
