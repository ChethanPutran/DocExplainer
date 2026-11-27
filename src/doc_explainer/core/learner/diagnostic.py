"""Choosing which concepts to test to find out where a new learner stands.

Each question should teach us as much as possible: so prefer concepts the text leans on heavily
(importance) and that we are most unsure about (belief near 50%), skip what we already know or
have just asked, and avoid asking about concepts that are direct prerequisites of each other, since
the answer to one already says a lot about the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from ..common.learner import Belief
from .tracker import ConceptGraph


@dataclass(frozen=True)
class Candidate:
    concept_id: str
    importance: int = 1


def informativeness(importance: int, p_known: float, max_importance: int) -> float:
    """High for important concepts whose status we are unsure about."""
    uncertainty = 1.0 - abs(2.0 * p_known - 1.0)
    return (importance / max(max_importance, 1)) * uncertainty


def plan_diagnostic(doc_id: str, candidates: Sequence[Candidate], beliefs: Dict[str, Belief], graph: ConceptGraph,
                    k: int = 6, already_asked: Sequence[str] = (), settled: float = 0.9) -> List[str]:
    """Up to ``k`` concept ids to ask about next, most informative first. Deterministic."""
    asked: Set[str] = set(already_asked)
    pool = [c for c in candidates if c.concept_id not in asked
            and not (beliefs.get(c.concept_id) and beliefs[c.concept_id].evidence >= 1 and
                     (beliefs[c.concept_id].p_known >= settled or beliefs[c.concept_id].p_known <= 1 - settled))]
    if not pool:
        return []
    top = max(c.importance for c in pool)
    scored = sorted(pool, key=lambda c: (-informativeness(c.importance, beliefs[c.concept_id].p_known if c.concept_id in beliefs else 0.5, top),
                                         c.concept_id))
    picked: List[str] = []
    blocked: Set[str] = set()
    for c in scored:
        if len(picked) >= k:
            break
        if c.concept_id in blocked:
            continue
        picked.append(c.concept_id)
        blocked.update(i for i, _ in graph.prerequisites_of(doc_id, c.concept_id))
        blocked.update(i for i, _ in graph.dependents_of(doc_id, c.concept_id))
    for c in scored:                                  # too few unrelated candidates: fill up with the rest
        if len(picked) >= k:
            break
        if c.concept_id not in picked:
            picked.append(c.concept_id)
    return picked
