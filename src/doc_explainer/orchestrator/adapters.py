"""The glue: each core package declares what it needs as a Protocol; these classes provide it from the real
implementations. This is the only module that knows two core packages at once."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.common.knowledge import Concept, RelationType
from ..core.common.learner import Belief, BeliefChange, Evidence, EvidenceKind, LearnerProfile
from ..core.common.structure import DocumentStructure
from ..core.indexing import JsonContentStore
from ..core.knowledge import KnowledgeView, SqliteKnowledgeStore, normalize_name
from ..core.learner import Candidate, LearnerTracker, classify_question, plan_diagnostic
from ..core.tutor import SectionConcepts

_WORD = re.compile(r"[\w\-]+")


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _tokens(text: str) -> List[str]:
    return [_stem(w) for w in _WORD.findall(text.lower())]


class StoreGraph:
    """How concepts depend on each other, for the learner model (reads the knowledge store)."""

    def __init__(self, store: SqliteKnowledgeStore):
        self.store = store

    def prerequisites_of(self, doc_id: str, concept_id: str) -> List[Tuple[str, float]]:
        return [(r.src, float(r.strength)) for r in self.store.relations(doc_id, type=RelationType.PREREQUISITE_OF) if r.dst == concept_id]

    def dependents_of(self, doc_id: str, concept_id: str) -> List[Tuple[str, float]]:
        return [(r.dst, float(r.strength)) for r in self.store.relations(doc_id, type=RelationType.PREREQUISITE_OF) if r.src == concept_id]


class KnowledgeAdapter:
    """The tutor's view of the knowledge graph."""

    def __init__(self, view: KnowledgeView):
        self.view = view
        self._phrases: Dict[str, Tuple[tuple, List[Tuple[Concept, List[List[str]]]]]] = {}

    def section_concepts(self, doc_id: str, structure: DocumentStructure, section_id: str) -> SectionConcepts:
        k = self.view.section_knowledge(doc_id, structure, section_id)
        return SectionConcepts(introduces=k.introduces, uses_earlier=k.uses_earlier)

    def concepts_in_text(self, doc_id: str, text: str, upto: int) -> List[Concept]:
        """Concepts (already met by ``upto``) whose name or alias appears in ``text`` as whole words."""
        toks = _tokens(text)
        found = []
        for concept, phrases in self._phrase_table(doc_id):
            if concept.introduced_ordinal > upto:
                continue
            for ph in phrases:
                n = len(ph)
                if n and any(toks[i:i + n] == ph for i in range(len(toks) - n + 1)):
                    found.append(concept)
                    break
        return sorted(found, key=lambda c: (-c.importance, c.introduced_ordinal))

    def _phrase_table(self, doc_id: str):
        concepts = self.view.store.concepts(doc_id)
        key = (len(concepts), sum(c.mention_count for c in concepts), sum(len(c.aliases) for c in concepts))
        cached = self._phrases.get(doc_id)
        if cached is None or cached[0] != key:
            table = [(c, [t for t in (_tokens(n) for n in (c.name, *c.aliases)) if t]) for c in concepts]
            self._phrases[doc_id] = cached = (key, table)
        return cached[1]

    def concept(self, doc_id: str, concept_id: str) -> Optional[Concept]:
        return self.view.store.concept(doc_id, concept_id)

    def concepts_upto(self, doc_id: str, ordinal: int) -> List[Concept]:
        return self.view.store.concepts(doc_id, ordinal)

    def prerequisites(self, doc_id: str, concept_id: str, depth: int, upto: int) -> List[Tuple[Concept, int]]:
        return [(p.concept, p.distance) for p in self.view.prerequisites(doc_id, concept_id, depth, upto)]

    def glossary(self, doc_id: str, ordinal: int, concept_ids: Sequence[str]) -> Dict[str, str]:
        return self.view.glossary(doc_id, ordinal, list(concept_ids))

    def diagnostic_candidates(self, doc_id: str, limit: int = 40) -> List[Concept]:
        """Background the text assumes, plus what it leans on most: the things worth asking about."""
        assumed = self.view.assumed(doc_id)
        central = sorted(self.view.store.concepts(doc_id), key=lambda c: (-c.importance, c.introduced_ordinal))
        seen, out = set(), []
        for c in [*assumed, *central]:
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
        return out[:limit]


class LearnerAdapter:
    """The tutor's view of the learner model: intents in, evidence out."""

    def __init__(self, tracker: LearnerTracker):
        self.tracker = tracker

    def beliefs(self, user_id: str, doc_id: str, concept_ids: Sequence[str]) -> Dict[str, Belief]:
        return self.tracker.beliefs(user_id, doc_id, list(concept_ids))

    def profile(self, user_id: str, doc_id: str) -> LearnerProfile:
        return self.tracker.profile(user_id, doc_id)

    def start(self, user_id, doc_id, level, known_concept_ids, onboarding) -> LearnerProfile:
        return self.tracker.start(user_id, doc_id, level, list(known_concept_ids), onboarding)

    def question_asked(self, user_id: str, doc_id: str, text: str, concept_ids: Sequence[str]) -> None:
        kind = classify_question(text)
        if kind and concept_ids:
            self.tracker.record(user_id, doc_id, Evidence(EvidenceKind.QUESTION_ASKED, list(concept_ids), question_type=kind))

    def explanation_feedback(self, user_id, doc_id, concept_ids, verdict) -> None:
        self.tracker.record(user_id, doc_id, Evidence(EvidenceKind.EXPLANATION_FEEDBACK, list(concept_ids), verdict=verdict))

    def read_section(self, user_id, doc_id, concept_ids) -> None:
        if concept_ids:
            self.tracker.record(user_id, doc_id, Evidence(EvidenceKind.READ_SECTION, list(concept_ids)))

    def quiz_answer(self, user_id, doc_id, concept_id, correct, difficulty) -> List[BeliefChange]:
        return self.tracker.record(user_id, doc_id, Evidence(EvidenceKind.QUIZ_ANSWER, [concept_id], correct=correct, difficulty=difficulty))

    def plan_diagnostic(self, user_id, doc_id, candidates, k, asked) -> List[str]:
        beliefs = self.tracker.store.get_beliefs(user_id, doc_id)
        return plan_diagnostic(doc_id, [Candidate(cid, imp) for cid, imp in candidates], beliefs, self.tracker.graph, k, asked)

    def recalibrate(self, user_id: str, doc_id: str) -> LearnerProfile:
        return self.tracker.recalibrate(user_id, doc_id)

    def position(self, user_id: str, doc_id: str) -> Optional[int]:
        return self.tracker.position(user_id, doc_id)

    def advance_position(self, user_id: str, doc_id: str, ordinal: int) -> int:
        return self.tracker.advance_position(user_id, doc_id, ordinal)


class StructureLoader:
    """The tutor's view of the stored documents (with a small cache: structures are read on every request)."""

    def __init__(self, content: JsonContentStore, capacity: int = 8):
        self.content, self.capacity = content, capacity
        self._cache: Dict[str, DocumentStructure] = {}

    def load(self, doc_id: str) -> DocumentStructure:
        if doc_id not in self._cache:
            if len(self._cache) >= self.capacity:
                self._cache.pop(next(iter(self._cache)))
            self._cache[doc_id] = self.content.load(doc_id)
        return self._cache[doc_id]

    def forget(self, doc_id: str) -> None:
        self._cache.pop(doc_id, None)
