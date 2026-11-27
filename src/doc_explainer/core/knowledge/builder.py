"""TemporalKnowledgeBuilder: grows a document's knowledge graph from empty, one section at a time.

For each step (see ``steps.plan_steps``) the LLM sees ONLY that section's text plus the concepts
already introduced. What it finds is resolved against the existing concepts, checked, and stored as a
``KnowledgeDelta`` stamped with the step's reading-order position. Rules enforced here, whatever the model says:

* nothing can depend on something the reader has not reached: a prerequisite edge may only point
  from a concept introduced earlier (or at the same point) to a later one;
* prerequisite edges never form a cycle;
* every mention is tied to a real paragraph of the step, and concepts are merged, not duplicated.

Builds are resumable (finished steps are skipped) and fail loudly: if the model keeps failing the build
stops with the reason instead of producing an empty graph.
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..common.knowledge import (Concept, KnowledgeDelta, Mention, MentionRole, Relation, RelationType, StepRecord)
from ..common.structure import DocumentStructure
from .canonical import Canonicalizer, normalize_name
from .extractor import ConceptExtractor
from .steps import Step, plan_steps
from .store import SqliteKnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeBuildError(RuntimeError):
    """The knowledge graph could not be built (the model keeps failing)."""


@dataclass
class BuildReport:
    steps_total: int = 0
    steps_done: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0            # already done in an earlier run
    concepts: int = 0
    relations: int = 0
    errors: List[str] = field(default_factory=list)


class TemporalKnowledgeBuilder:
    def __init__(self, extractor: ConceptExtractor, canonicalizer: Canonicalizer, store: SqliteKnowledgeStore,
                 max_consecutive_failures: int = 3, max_failure_ratio: float = 0.3):
        self.extractor, self.canon, self.store = extractor, canonicalizer, store
        self.max_consecutive_failures, self.max_failure_ratio = max_consecutive_failures, max_failure_ratio

    # ------------------------------------------------------------------ whole document
    def build(self, structure: DocumentStructure, progress: Optional[Callable[[str, float], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None) -> BuildReport:
        doc_id = structure.doc_id
        steps = plan_steps(structure)
        signature = hashlib.sha256("\x1f".join(f"{s.index}|{s.section_id}|{t}" for s in steps for _, _, t in s.paragraphs).encode()).hexdigest()
        if self.store.get_meta(doc_id, "plan") != signature:     # the document changed: start over
            self.store.reset(doc_id)
            self.store.set_meta(doc_id, "plan", signature)
        done = {r.step for r in self.store.steps(doc_id) if r.status == "done"}
        unfinished = [s.index for s in steps if s.index not in done]
        if unfinished and any(d > unfinished[0] for d in done):
            # a middle step is being redone: forget everything after it so the graph stays in reading order
            self.store.rewind(doc_id, unfinished[0])
            done = {r.step for r in self.store.steps(doc_id) if r.status == "done"}
        self.canon.load(self.store.concepts(doc_id))
        report = BuildReport(steps_total=len(steps), steps_skipped=len(done))
        consecutive = 0
        for i, step in enumerate(steps):
            if progress:
                progress(f"reading: {step.path or step.title}", i / max(len(steps), 1))
            if step.index in done:
                continue
            if cancelled and cancelled():
                break
            try:
                self.advance(doc_id, step)
                report.steps_done += 1
                consecutive = 0
            except Exception as e:                                # noqa: BLE001 - recorded, then decided below
                message = f"{type(e).__name__}: {e}"
                logger.warning("Knowledge step %d (%s) failed: %s", step.index, step.path, message)
                self.store.record_failure(doc_id, StepRecord(step.index, step.section_id, step.ordinal_start,
                                                             step.ordinal_end, step.words, "failed", error=message[:500]))
                report.steps_failed += 1
                report.errors.append(message)
                consecutive += 1
                if consecutive >= self.max_consecutive_failures:
                    raise KnowledgeBuildError(
                        f"The model failed {consecutive} steps in a row; stopping. Last error: {message}") from e
        if progress:
            progress("knowledge built", 1.0)
        report.concepts = len(self.store.concepts(doc_id))
        report.relations = len(self.store.relations(doc_id))
        attempted = report.steps_done + report.steps_failed
        if attempted and report.steps_failed / attempted > self.max_failure_ratio:
            raise KnowledgeBuildError(f"{report.steps_failed} of {attempted} steps failed. Last error: {report.errors[-1]}")
        return report

    # ------------------------------------------------------------------ one step
    def advance(self, doc_id: str, step: Step) -> KnowledgeDelta:
        known = self.store.concepts(doc_id)
        extraction = self.extractor.extract(step, known)
        delta = KnowledgeDelta(step=step.index, section_id=step.section_id, ordinal_start=step.ordinal_start,
                               ordinal_end=step.ordinal_end, words=step.words)
        by_id: Dict[str, Concept] = {c.id: c for c in known}
        next_n = self.store.next_concept_number(doc_id)
        resolved: Dict[str, str] = {}
        seen_mentions: Set[Tuple[str, str]] = set()
        importance: Counter = Counter()

        def where(p: int) -> Tuple[str, int]:
            node_id, ordinal, _ = step.paragraphs[min(max(p, 1), len(step.paragraphs)) - 1]
            return node_id, ordinal

        for rc in extraction.concepts:
            first_node, first_ordinal = where(rc.paragraphs[0] if rc.paragraphs else 1)
            cid = rc.known_id if rc.known_id in by_id else None
            if cid is None:
                cid = self.canon.match([rc.name, *rc.aliases])
            if cid is None:
                concept = Concept(id=f"c{next_n}", name=rc.name, introduced_step=step.index, introduced_ordinal=first_ordinal,
                                  first_node_id=first_node,
                                  aliases=[a for a in rc.aliases if normalize_name(a) != normalize_name(rc.name)])
                next_n += 1
                delta.new_concepts.append(concept)
                by_id[concept.id] = concept
                self.canon.add(concept)
                cid = concept.id
            else:
                existing = by_id[cid]
                have = {normalize_name(existing.name), *(normalize_name(a) for a in existing.aliases)}
                fresh = [n for n in dict.fromkeys([rc.name, *rc.aliases]) if normalize_name(n) not in have]
                if fresh:
                    delta.updated_aliases.append((cid, fresh))
                    existing.aliases = existing.aliases + fresh
                    for n in fresh:
                        self.canon.add_alias(cid, n)
            for n in (rc.name, *rc.aliases):
                resolved[normalize_name(n)] = cid
            for p in rc.paragraphs or [1]:
                node_id, ordinal = where(p)
                if (cid, node_id) not in seen_mentions:
                    seen_mentions.add((cid, node_id))
                    delta.mentions.append(Mention(cid, node_id, ordinal, step.index, rc.role))
            if rc.role is MentionRole.DEFINES and rc.definition:
                node_id, ordinal = where(rc.paragraphs[0] if rc.paragraphs else 1)
                delta.definitions.append((cid, ordinal, rc.definition, node_id))
            importance[cid] += rc.importance
        delta.importance = sorted(importance.items())

        self._relations(doc_id, step, extraction, delta, by_id, resolved, where)
        record = StepRecord(
            step=step.index, section_id=step.section_id, ordinal_start=step.ordinal_start, ordinal_end=step.ordinal_end,
            words=step.words, status="done", new_concepts=len(delta.new_concepts), new_relations=len(delta.new_relations),
            mentions=len(delta.mentions), known_mentions=sum(1 for m in delta.mentions if by_id[m.concept_id].introduced_step < step.index))
        self.store.apply(doc_id, delta, record)
        return delta

    def _relations(self, doc_id, step, extraction, delta, by_id, resolved, where) -> None:
        existing = {r.key for r in self.store.relations(doc_id)}
        prereq: Dict[str, Set[str]] = {}
        for r in self.store.relations(doc_id, type=RelationType.PREREQUISITE_OF):
            prereq.setdefault(r.src, set()).add(r.dst)

        def reaches(start: str, goal: str) -> bool:
            stack, seen = [start], set()
            while stack:
                n = stack.pop()
                if n == goal:
                    return True
                if n not in seen:
                    seen.add(n)
                    stack.extend(prereq.get(n, ()))
            return False

        def lookup(name: str) -> Optional[str]:
            return resolved.get(normalize_name(name)) or self.canon.match([name])

        added: Dict[Tuple[str, str, str], Relation] = {}
        for rr in extraction.relations:
            src, dst = lookup(rr.source), lookup(rr.target)
            if not src or not dst:
                delta.rejected.append(f"unknown concept in relation '{rr.source}' -> '{rr.target}'")
                continue
            if src == dst:
                delta.rejected.append(f"self relation on '{rr.source}'")
                continue
            node_id, ordinal = where(rr.paragraph or 1)
            if rr.type.symmetric:                                  # one record per pair, earliest concept first
                a, b = sorted((src, dst), key=lambda i: (by_id[i].introduced_ordinal, i))
                src, dst = a, b
            if rr.type is RelationType.PREREQUISITE_OF:
                if by_id[src].introduced_ordinal > by_id[dst].introduced_ordinal:
                    delta.rejected.append(f"forward dependency: '{by_id[src].name}' appears after '{by_id[dst].name}'")
                    continue
                if reaches(dst, src):
                    delta.rejected.append(f"cycle: '{by_id[src].name}' -> '{by_id[dst].name}'")
                    continue
            key = (src, dst, rr.type.value)
            if key in existing:
                delta.strengthened.append((src, dst, rr.type.value, node_id))
            elif key in added:
                added[key].strength += 1
            else:
                added[key] = Relation(src, dst, rr.type, first_ordinal=ordinal, first_step=step.index, evidence=[node_id])
                if rr.type is RelationType.PREREQUISITE_OF:
                    prereq.setdefault(src, set()).add(dst)
        delta.new_relations.extend(added.values())
