"""Read-side questions about a document's knowledge graph. Pure queries over the store; nothing is written.

Every question can be asked "as of" a reading position (an ordinal): the graph is then exactly what a reader
who has got that far has met. Section dependencies only ever point at EARLIER sections.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..common.knowledge import Concept, Relation, RelationType
from ..common.structure import DocumentStructure
from .canonical import normalize_name
from .store import SqliteKnowledgeStore


@dataclass
class GraphSnapshot:
    as_of: Optional[int]
    concepts: List[Concept]
    relations: List[Relation]


@dataclass
class SectionKnowledge:
    section_id: str
    introduces: List[Concept] = field(default_factory=list)      # first appear in this section's own content
    defines: List[Concept] = field(default_factory=list)         # explained here (may have appeared earlier)
    uses_earlier: List[Concept] = field(default_factory=list)    # relied on here, introduced before this section


@dataclass
class SectionDependency:
    section_id: str                    # an EARLIER section
    title: str
    concept_ids: List[str]
    weight: int


@dataclass
class GrowthPoint:
    step: int
    section_id: str
    ordinal_end: int
    words: int
    new_concepts: int
    total_concepts: int
    total_relations: int
    novelty: float                     # new concepts per 1,000 words
    reuse: float                       # share of mentions that are of concepts met before


@dataclass
class PrerequisiteItem:
    concept: Concept
    distance: int
    strength: int


class KnowledgeView:
    def __init__(self, store: SqliteKnowledgeStore):
        self.store = store

    # ------------------------------------------------------------------ the graph, at a point in the document
    def graph_at(self, doc_id: str, ordinal: Optional[int] = None) -> GraphSnapshot:
        concepts = self.store.concepts(doc_id, ordinal)
        ids = {c.id for c in concepts}
        relations = [r for r in self.store.relations(doc_id, ordinal) if r.src in ids and r.dst in ids]
        return GraphSnapshot(ordinal, concepts, relations)

    def find(self, doc_id: str, name: str) -> Optional[Concept]:
        wanted = normalize_name(name)
        for c in self.store.concepts(doc_id):
            if wanted in {normalize_name(c.name), *(normalize_name(a) for a in c.aliases)}:
                return c
        return None

    # ------------------------------------------------------------------ sections
    def section_knowledge(self, doc_id: str, structure: DocumentStructure, section_id: str) -> SectionKnowledge:
        section = structure.node(section_id)
        lo, hi = section.ordinal, structure.own_end(section_id)
        concepts = {c.id: c for c in self.store.concepts(doc_id)}
        out = SectionKnowledge(section_id)
        out.introduces = [c for c in concepts.values() if lo <= c.introduced_ordinal <= hi]
        out.defines = [c for c in concepts.values() if c.defined_ordinal is not None and lo <= c.defined_ordinal <= hi]
        used = {m.concept_id for m in self.store.mentions(doc_id, lo=lo, hi=hi)}
        out.uses_earlier = sorted((concepts[i] for i in used if concepts[i].introduced_ordinal < lo),
                                  key=lambda c: -c.importance)
        out.introduces.sort(key=lambda c: -c.importance)
        return out

    def section_dependencies(self, doc_id: str, structure: DocumentStructure) -> Dict[str, List[SectionDependency]]:
        """For each section: the earlier sections whose concepts it relies on, strongest first."""
        concepts = {c.id: c for c in self.store.concepts(doc_id)}
        origin = {}
        for c in concepts.values():
            if c.first_node_id in structure:
                sec = structure.section_of(c.first_node_id)
                if sec is not None:
                    origin[c.id] = sec
        result: Dict[str, List[SectionDependency]] = {}
        for section in structure.sections():
            lo, hi = section.ordinal, structure.own_end(section.id)
            by_origin: Dict[str, List[str]] = defaultdict(list)
            for cid in sorted({m.concept_id for m in self.store.mentions(doc_id, lo=lo, hi=hi)}):
                c = concepts[cid]
                if c.introduced_ordinal < lo and cid in origin and origin[cid].id != section.id:
                    by_origin[origin[cid].id].append(cid)
            deps = [SectionDependency(sid, structure.node(sid).title, ids, len(ids)) for sid, ids in by_origin.items()]
            result[section.id] = sorted(deps, key=lambda d: (-d.weight, structure.node(d.section_id).ordinal))
        return result

    # ------------------------------------------------------------------ how the document's knowledge grows
    def growth(self, doc_id: str) -> List[GrowthPoint]:
        points, total_c = [], 0
        relations = self.store.relations(doc_id)
        for r in self.store.steps(doc_id):
            if r.status != "done":
                continue
            total_c += r.new_concepts
            points.append(GrowthPoint(
                step=r.step, section_id=r.section_id, ordinal_end=r.ordinal_end, words=r.words, new_concepts=r.new_concepts,
                total_concepts=total_c, total_relations=sum(1 for x in relations if x.first_step <= r.step),
                novelty=round(r.new_concepts / r.words * 1000, 2) if r.words else 0.0,
                reuse=round(r.known_mentions / r.mentions, 2) if r.mentions else 0.0))
        return points

    # ------------------------------------------------------------------ dependencies between concepts
    def prerequisites(self, doc_id: str, concept_id: str, depth: int = 3, upto: Optional[int] = None) -> List[PrerequisiteItem]:
        concepts = {c.id: c for c in self.store.concepts(doc_id, upto)}
        incoming: Dict[str, List[Relation]] = defaultdict(list)
        for r in self.store.relations(doc_id, upto, RelationType.PREREQUISITE_OF):
            incoming[r.dst].append(r)
        out, seen, frontier = [], {concept_id}, [concept_id]
        for distance in range(1, depth + 1):
            nxt = []
            for node in frontier:
                for r in incoming.get(node, []):
                    if r.src not in seen and r.src in concepts:
                        seen.add(r.src)
                        nxt.append(r.src)
                        out.append(PrerequisiteItem(concepts[r.src], distance, r.strength))
            frontier = nxt
        return sorted(out, key=lambda p: (p.distance, -p.strength, p.concept.introduced_ordinal))

    def learning_path(self, doc_id: str, target_id: str, upto: Optional[int] = None) -> List[Concept]:
        """Everything needed to understand ``target``, in the order the document teaches it, ending with it.

        Prerequisite edges only ever point earlier -> later, so reading order is a valid topological order."""
        target = self.store.concept(doc_id, target_id)
        if target is None:
            return []
        needed = [p.concept for p in self.prerequisites(doc_id, target_id, depth=1000, upto=upto)]
        return sorted(needed, key=lambda c: (c.introduced_ordinal, c.id)) + [target]

    def related(self, doc_id: str, concept_id: str, limit: int = 8, upto: Optional[int] = None) -> List[Tuple[Concept, Relation]]:
        concepts = {c.id: c for c in self.store.concepts(doc_id, upto)}
        hits = []
        for r in self.store.relations(doc_id, upto):
            other = r.dst if r.src == concept_id else r.src if r.dst == concept_id else None
            if other and other in concepts:
                hits.append((concepts[other], r))
        return sorted(hits, key=lambda h: (-h[1].strength, h[0].introduced_ordinal))[:limit]

    # ------------------------------------------------------------------ knowledge the text assumes vs. teaches
    def assumed(self, doc_id: str) -> List[Concept]:
        """Concepts the document relies on but never defines: background the reader is expected to bring."""
        return sorted((c for c in self.store.concepts(doc_id) if c.defined_ordinal is None),
                      key=lambda c: (-c.importance, c.introduced_ordinal))

    def glossary(self, doc_id: str, ordinal: int, concept_ids: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Concept id -> its definition as the text had given it by ``ordinal`` (the latest one so far)."""
        wanted = set(concept_ids) if concept_ids is not None else None
        latest: Dict[str, str] = {}
        for cid, at, text, _ in self.store.definitions(doc_id):
            if at <= ordinal and (wanted is None or cid in wanted):
                latest[cid] = text
        return latest
