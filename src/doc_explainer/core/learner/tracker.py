"""LearnerTracker: turns what a learner does into a belief about each concept, and keeps those beliefs.

Evidence comes from several places (graded answers, the kind of questions asked, feedback on explanations,
reading, self reports). Each updates beliefs by its own rule (see ``evidence``), and the concept graph
spreads the news: doing well on a concept says something about what it rests on, doing badly says
something about what is built on it. Beliefs fade with time. Every change is logged.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from ..common.learner import (Belief, BeliefChange, Evidence, EvidenceKind, LearnerProfile, level_for)
from . import bkt
from . import evidence as ev
from .store import SqliteLearnerStore

LEVEL_PRIORS = {"beginner": 0.15, "intermediate": 0.35, "advanced": 0.60, "expert": 0.80}
KNOWN_TOPIC_BELIEF = 0.85
KNOWN_TOPIC_PREREQ_BELIEF = 0.70
GUESS_RATE = 0.25            # a four-option question answered at random
DAY = 86400.0


class ConceptGraph(Protocol):
    """What the tracker needs to know about how concepts depend on each other (supplied by the orchestrator)."""

    def prerequisites_of(self, doc_id: str, concept_id: str) -> List[Tuple[str, float]]: ...

    def dependents_of(self, doc_id: str, concept_id: str) -> List[Tuple[str, float]]: ...


class LearnerTracker:
    def __init__(self, store: SqliteLearnerStore, graph: ConceptGraph, params: bkt.BKTParams = bkt.BKTParams(),
                 clock: Callable[[], float] = time.time):
        self.store, self.graph, self.params, self.clock = store, graph, params, clock

    # ------------------------------------------------------------------ onboarding
    def start(self, user_id: str, doc_id: str, level: str = "intermediate", known_concept_ids: Sequence[str] = (),
              onboarding: Optional[dict] = None) -> LearnerProfile:
        """Set the learner's starting point: a level (the prior for every concept) and topics they say they know."""
        if level not in LEVEL_PRIORS:
            raise ValueError(f"level must be one of {sorted(LEVEL_PRIORS)}")
        now = self.clock()
        profile = LearnerProfile(user_id, doc_id, LEVEL_PRIORS[level], level, dict(onboarding or {}, level=level), now)
        self.store.save_profile(profile)
        beliefs: Dict[str, Belief] = {}
        for cid in known_concept_ids:
            beliefs[cid] = Belief(max(KNOWN_TOPIC_BELIEF, profile.prior), 1, now)
            for pre, _ in self.graph.prerequisites_of(doc_id, cid):
                if pre not in known_concept_ids:
                    beliefs[pre] = Belief(max(KNOWN_TOPIC_PREREQ_BELIEF, profile.prior), 1, now)
        if beliefs:
            self.store.set_beliefs(user_id, doc_id, beliefs)
        self.store.add_events(user_id, doc_id, [(now, "onboarding", None, json.dumps(profile.onboarding), None, None)])
        return profile

    def profile(self, user_id: str, doc_id: str) -> LearnerProfile:
        return self.store.get_profile(user_id, doc_id) or LearnerProfile(user_id, doc_id, LEVEL_PRIORS["intermediate"], "intermediate")

    # ------------------------------------------------------------------ reading beliefs
    def beliefs(self, user_id: str, doc_id: str, concept_ids: Optional[Iterable[str]] = None) -> Dict[str, Belief]:
        """Current beliefs with forgetting applied. Concepts with no evidence get the learner's prior."""
        profile = self.profile(user_id, doc_id)
        ids = list(concept_ids) if concept_ids is not None else None
        stored = self.store.get_beliefs(user_id, doc_id, ids)
        now = self.clock()
        out: Dict[str, Belief] = {}
        for cid in (ids if ids is not None else list(stored)):
            b = stored.get(cid)
            if b is None:
                out[cid] = Belief(profile.prior, 0, 0.0)
            else:
                faded = bkt.decay(b.p_known, profile.prior, (now - b.updated_at) / DAY, b.evidence, self.params)
                out[cid] = Belief(faded, b.evidence, b.updated_at)
        return out

    def belief(self, user_id: str, doc_id: str, concept_id: str) -> Belief:
        return self.beliefs(user_id, doc_id, [concept_id])[concept_id]

    # ------------------------------------------------------------------ evidence in
    def record(self, user_id: str, doc_id: str, e: Evidence) -> List[BeliefChange]:
        """Apply one piece of evidence; returns the beliefs that changed."""
        updates: Dict[str, Tuple[float, bool]] = {}      # concept -> (new p, counts as evidence)
        current = {c: self.beliefs(user_id, doc_id, [c])[c].p_known for c in e.concept_ids}

        def p_of(cid: str) -> float:
            if cid in updates:
                return updates[cid][0]
            if cid not in current:
                current[cid] = self.beliefs(user_id, doc_id, [cid])[cid].p_known
            return current[cid]

        def lift(cid: str, correct: bool, weight: float, counts: bool = True, difficulty: float = 0.5, only: str = "") -> None:
            new = bkt.observe(p_of(cid), correct, self.params, difficulty, weight)
            if only == "up":
                new = max(new, p_of(cid))
            elif only == "down":
                new = min(new, p_of(cid))
            updates[cid] = (new, counts or (cid in updates and updates[cid][1]))

        if e.kind is EvidenceKind.QUIZ_ANSWER and e.correct is not None:
            for cid in e.concept_ids:
                lift(cid, e.correct, 1.0, difficulty=e.difficulty)
                if e.correct:
                    for pre, _ in self.graph.prerequisites_of(doc_id, cid):
                        if pre not in e.concept_ids:
                            lift(pre, True, ev.WEIGHT_PROPAGATE_UP, only="up")
                else:
                    for dep, _ in self.graph.dependents_of(doc_id, cid):
                        if dep not in e.concept_ids:
                            lift(dep, False, ev.WEIGHT_PROPAGATE_DOWN, only="down")
        elif e.kind is EvidenceKind.QUESTION_ASKED:
            if e.question_type == "definition":
                for cid in e.concept_ids:
                    lift(cid, False, ev.WEIGHT_ASKED_DEFINITION)
            elif e.question_type == "deeper":
                for cid in e.concept_ids:
                    lift(cid, True, ev.WEIGHT_ASKED_DEEPER)
                    for pre, _ in self.graph.prerequisites_of(doc_id, cid):
                        lift(pre, True, ev.WEIGHT_ASKED_DEEPER, only="up")
        elif e.kind is EvidenceKind.EXPLANATION_FEEDBACK:
            if e.verdict == "too_easy":
                for cid in e.concept_ids:
                    lift(cid, True, ev.WEIGHT_TOO_EASY, only="up")
            elif e.verdict == "too_hard":
                for cid in e.concept_ids:
                    lift(cid, False, ev.WEIGHT_TOO_HARD, only="down")
        elif e.kind is EvidenceKind.READ_SECTION:
            for cid in e.concept_ids:
                updates[cid] = (bkt.exposure(p_of(cid), ev.EXPOSURE_AMOUNT), False)
        elif e.kind is EvidenceKind.SELF_REPORT and e.value is not None:
            for cid in e.concept_ids:
                updates[cid] = (min(max(0.4 * p_of(cid) + 0.6 * e.value, 0.01), 0.99), True)
        return self._commit(user_id, doc_id, e, updates)

    def _commit(self, user_id: str, doc_id: str, e: Evidence, updates: Dict[str, Tuple[float, bool]]) -> List[BeliefChange]:
        if not updates:
            return []
        now = self.clock()
        stored = self.store.get_beliefs(user_id, doc_id, list(updates))
        profile = self.profile(user_id, doc_id)
        changes, new_beliefs, rows = [], {}, []
        payload = json.dumps({"concepts": e.concept_ids, "correct": e.correct, "difficulty": e.difficulty,
                              "question_type": e.question_type, "verdict": e.verdict, **e.payload})
        for cid, (p, counts) in updates.items():
            old = stored.get(cid)
            before = self.beliefs(user_id, doc_id, [cid])[cid].p_known
            new_beliefs[cid] = Belief(p, (old.evidence if old else 0) + (1 if counts else 0), now)
            changes.append(BeliefChange(cid, before, p))
            rows.append((now, e.kind.value, cid, payload, before, p))
        self.store.set_beliefs(user_id, doc_id, new_beliefs)
        self.store.add_events(user_id, doc_id, rows)
        return changes

    # ------------------------------------------------------------------ after a diagnostic
    def recalibrate(self, user_id: str, doc_id: str) -> LearnerProfile:
        """Set the starting point from how the learner actually did on graded questions (the diagnostic).

        Per-concept beliefs move slowly (one answer per concept), so they are the wrong tool for judging overall
        level from a handful of questions on different concepts. Instead: the share answered correctly, corrected
        for lucky guesses (4 options -> 25%), smoothed, and blended with their own account of their level, which
        counts like one extra question. Concepts we know nothing about then start from this prior."""
        profile = self.profile(user_id, doc_id)
        graded = [e["payload"]["correct"] for e in self.store.events(user_id, doc_id, limit=100_000)
                  if e["kind"] == EvidenceKind.QUIZ_ANSWER.value and e["concept_id"] in e["payload"].get("concepts", [])
                  and e["payload"].get("correct") is not None]
        if graded:
            n = len(graded)
            accuracy = sum(1 for c in graded if c) / n
            corrected = max(0.0, (accuracy - GUESS_RATE) / (1 - GUESS_RATE))
            estimate = (n * corrected + 1) / (n + 2)
            profile.prior = min(max((n * estimate + profile.prior) / (n + 1), 0.05), 0.95)
            profile.level = level_for(profile.prior)
            self.store.save_profile(profile)
        return profile

    # ------------------------------------------------------------------ where the reader is
    def advance_position(self, user_id: str, doc_id: str, ordinal: int) -> int:
        """Remember the furthest point reached (never moves backwards)."""
        best = max(ordinal, self.store.get_position(user_id, doc_id) or 0)
        self.store.set_position(user_id, doc_id, best, self.clock())
        return best

    def position(self, user_id: str, doc_id: str) -> Optional[int]:
        return self.store.get_position(user_id, doc_id)

    def forget(self, user_id: str, doc_id: Optional[str] = None) -> None:
        self.store.delete(user_id, doc_id)
