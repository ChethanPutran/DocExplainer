"""The knowledge / learner / quiz side of the API: structure, the knowledge graph over time, the learner's state,
quizzes and the diagnostic. A thin layer: it checks ownership, calls the ``System`` and shapes the results."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from doc_explainer.core.common.knowledge import Concept

from .errors import ServiceError
from .schemas import (Ack, AnswerOut, BeliefOut, ConceptOut, ConceptResult, ContextOut, DiagnosticBody, DiagnosticOut, FeedbackBody,
                      GrowthOut, GrowthPointOut, HistoryOut, HistoryPointOut, KnowledgeOut, ReviewItemOut, ReviewOut, LearnerOut, QuizOut, QuizQuestionOut, RelationOut, SectionDependencyOut,
                      SectionKnowledgeOut, SectionOut, StructureOut)

MAX_LEARNER_CONCEPTS = 60
REVIEW_THRESHOLD = 0.6          # retention below this: due for review
HALF_LIFE_DAYS = 2.0            # memory half-life with one observation; each further observation doubles it, to a cap
MAX_HALF_LIFE_DAYS = 60.0


def retention(p_known: float, evidence: int, age_days: float) -> float:
    """What the reader probably still remembers: the belief decayed with time since it was last practised. More evidence
    means slower forgetting (a simple spaced-repetition curve)."""
    half_life = min(MAX_HALF_LIFE_DAYS, HALF_LIFE_DAYS * (2 ** max(0, evidence - 1)))
    return p_known * 0.5 ** (max(0.0, age_days) / half_life)


def _concept_out(c: Concept, structure) -> ConceptOut:
    section = structure.section_of(c.first_node_id) if c.first_node_id in structure else None
    return ConceptOut(id=c.id, name=c.name, introduced_ordinal=c.introduced_ordinal, section_id=section.id if section else "",
                      definition=c.definition, importance=c.importance, aliases=list(c.aliases))


class LearningApi:
    def __init__(self, system: Any, owned: Callable[[str, str], dict]):
        self._sys, self._owned = system, owned

    # ------------------------------------------------------------------ the document
    def structure(self, user_id: str, doc_id: str) -> StructureOut:
        self._owned(user_id, doc_id)
        s = self._sys.structure(doc_id)
        return StructureOut(doc_id=doc_id, title=s.title, format=s.source_format, pages=s.page_count, sections=[
            SectionOut(id=n.id, ordinal=n.ordinal, title=n.title, depth=n.depth, parent_id=n.parent_id if n.parent_id != "doc" else None,
                       page_start=n.page_start, page_end=n.page_end) for n in s.sections()])

    def knowledge(self, user_id: str, doc_id: str, upto: Optional[int]) -> KnowledgeOut:
        """The knowledge graph as a reader at position ``upto`` would know it (whole document if omitted)."""
        self._owned(user_id, doc_id)
        s = self._sys.structure(doc_id)
        g = self._sys.view.graph_at(doc_id, upto)
        return KnowledgeOut(as_of=upto, concepts=[_concept_out(c, s) for c in g.concepts],
                            relations=[RelationOut(src=r.src, dst=r.dst, type=r.type.value, strength=r.strength) for r in g.relations])

    def section_knowledge(self, user_id: str, doc_id: str, section_ordinal: int) -> SectionKnowledgeOut:
        self._owned(user_id, doc_id)
        s = self._sys.structure(doc_id)
        section = s.section_by_ordinal(section_ordinal)
        if section is None:
            raise ServiceError("invalid_params", f"{section_ordinal} is not the position of a section.")
        k = self._sys.view.section_knowledge(doc_id, s, section.id)
        names = {c.id: c.name for c in self._sys.knowledge_store.concepts(doc_id)}
        deps = self._sys.view.section_dependencies(doc_id, s).get(section.id, [])
        return SectionKnowledgeOut(
            section_id=section.id, title=section.title,
            introduces=[_concept_out(c, s) for c in k.introduces], uses_earlier=[_concept_out(c, s) for c in k.uses_earlier],
            depends_on=[SectionDependencyOut(section_id=d.section_id, title=d.title, concepts=[names.get(i, i) for i in d.concept_ids], weight=d.weight)
                        for d in deps])

    def growth(self, user_id: str, doc_id: str) -> GrowthOut:
        self._owned(user_id, doc_id)
        return GrowthOut(points=[GrowthPointOut(step=p.step, section_id=p.section_id, ordinal_end=p.ordinal_end, words=p.words,
                                                new_concepts=p.new_concepts, total_concepts=p.total_concepts,
                                                total_relations=p.total_relations, novelty=p.novelty, reuse=p.reuse)
                                 for p in self._sys.view.growth(doc_id)])

    # ------------------------------------------------------------------ concepts
    def _concept(self, doc_id: str, name: str) -> Concept:
        c = self._sys.view.find(doc_id, name)
        if c is None:
            raise ServiceError("concept_not_found", f"No concept called {name!r} in this document.")
        return c

    def prerequisites(self, user_id: str, doc_id: str, name: str, depth: int = 3) -> ConceptResult:
        self._owned(user_id, doc_id)
        c = self._concept(doc_id, name)
        return ConceptResult(concept=c.name, data=[{"name": p.concept.name, "distance": p.distance, "strength": p.strength}
                                                   for p in self._sys.view.prerequisites(doc_id, c.id, depth)])

    def learning_path(self, user_id: str, doc_id: str, name: str) -> ConceptResult:
        """What to understand, in the order the document teaches it, to understand this concept."""
        self._owned(user_id, doc_id)
        c = self._concept(doc_id, name)
        return ConceptResult(concept=c.name, data=[{"name": x.name, "definition": x.definition, "ordinal": x.introduced_ordinal}
                                                   for x in self._sys.view.learning_path(doc_id, c.id)])

    def recommendations(self, user_id: str, doc_id: str, name: str, limit: int = 5) -> ConceptResult:
        self._owned(user_id, doc_id)
        c = self._concept(doc_id, name)
        return ConceptResult(concept=c.name, data=[{"name": o.name, "relation": r.type.value, "strength": r.strength}
                                                   for o, r in self._sys.view.related(doc_id, c.id, limit)])

    # ------------------------------------------------------------------ the learner
    def learner(self, user_id: str, doc_id: str) -> LearnerOut:
        self._owned(user_id, doc_id)
        t = self._sys.tracker
        profile, position = t.profile(user_id, doc_id), t.position(user_id, doc_id)
        concepts = sorted(self._sys.knowledge_store.concepts(doc_id, position), key=lambda c: -c.importance)[:MAX_LEARNER_CONCEPTS]
        beliefs = t.beliefs(user_id, doc_id, [c.id for c in concepts])
        return LearnerOut(level=profile.level, prior=round(profile.prior, 3), position=position,
                          concepts=[BeliefOut(concept_id=c.id, name=c.name, p_known=round(beliefs[c.id].p_known, 3), evidence=beliefs[c.id].evidence)
                                    for c in concepts])

    def review(self, user_id: str, doc_id: str, limit: int = 20, now_ts: Optional[float] = None) -> ReviewOut:
        """Concepts the reader has met and is likely to be forgetting, weakest first (spaced repetition)."""
        import time
        self._owned(user_id, doc_id)
        now_ts = time.time() if now_ts is None else now_ts
        t = self._sys.tracker
        position = t.position(user_id, doc_id)
        concepts = {c.id: c for c in self._sys.knowledge_store.concepts(doc_id, position)}
        items = []
        for cid, b in t.beliefs(user_id, doc_id, list(concepts)).items():
            if b.evidence <= 0 or not b.updated_at:
                continue                                       # never practised: nothing to forget yet
            r = retention(b.p_known, b.evidence, (now_ts - b.updated_at) / 86400.0)
            items.append(ReviewItemOut(concept_id=cid, name=concepts[cid].name, p_known=round(b.p_known, 3), retention=round(r, 3),
                                       last_seen=b.updated_at, due=r < REVIEW_THRESHOLD))
        items.sort(key=lambda i: i.retention)
        return ReviewOut(items=items[:limit])

    def history(self, user_id: str, doc_id: str, limit: int = 200) -> HistoryOut:
        """How the reader's belief in each concept changed, in order: the data behind a progress chart."""
        self._owned(user_id, doc_id)
        events = self._sys.learner_store.events(user_id, doc_id, limit)
        names = {c.id: c.name for c in self._sys.knowledge_store.concepts(doc_id, None)}
        return HistoryOut(points=[HistoryPointOut(at=e["at"], kind=e["kind"], concept_id=e["concept_id"] or "",
                                                  name=names.get(e["concept_id"], ""), before=e["before"] or 0.0, after=e["after"] or 0.0)
                                  for e in events if e.get("concept_id")])

    def context(self, user_id: str, doc_id: str, section_ordinal: int) -> ContextOut:
        """What is known about this place in the document, and about this reader there."""
        s = self._sys.structure(doc_id)
        section, position = self._sys.tutor.locate(s, section_ordinal, user_id)
        if section.kind.value != "section":
            return ContextOut(session_data={"position": position})
        k = self._sys.view.section_knowledge(doc_id, s, section.id)
        concepts = [*k.introduces, *k.uses_earlier]
        beliefs = self._sys.tracker.beliefs(user_id, doc_id, [c.id for c in concepts])
        graph = self._sys.view.graph_at(doc_id, position)
        return ContextOut(
            user_knowledge={c.name: round(beliefs[c.id].p_known, 3) for c in concepts},
            session_data={"position": position, "level": self._sys.tracker.profile(user_id, doc_id).level,
                          "reading_position": self._sys.tracker.position(user_id, doc_id)},
            document_context={"section": " > ".join(s.section_path(section.id)), "page": section.page_start,
                              "introduces": [c.name for c in k.introduces], "uses_earlier": [c.name for c in k.uses_earlier]},
            concept_graph_info={"concepts_so_far": len(graph.concepts), "relations_so_far": len(graph.relations)})

    def start_diagnostic(self, user_id: str, doc_id: str, body: DiagnosticBody) -> DiagnosticOut:
        self._owned(user_id, doc_id)
        profile, questions = self._sys.quiz.start_diagnostic(user_id, doc_id, body.level, body.known_topics, body.questions)
        return DiagnosticOut(level=profile.level, prior=round(profile.prior, 3), questions=[self._q(q) for q in questions])

    def finish_diagnostic(self, user_id: str, doc_id: str) -> DiagnosticOut:
        self._owned(user_id, doc_id)
        profile = self._sys.quiz.finish_diagnostic(user_id, doc_id)
        return DiagnosticOut(level=profile.level, prior=round(profile.prior, 3), questions=[])

    def quiz(self, user_id: str, doc_id: str, section_ordinal: Optional[int], count: int) -> QuizOut:
        self._owned(user_id, doc_id)
        s = self._sys.structure(doc_id)
        _, position = self._sys.tutor.locate(s, section_ordinal, user_id)
        return QuizOut(questions=[self._q(q) for q in self._sys.quiz.practice(user_id, doc_id, position, count)])

    def answer(self, user_id: str, doc_id: str, question_id: str, choice: int) -> AnswerOut:
        self._owned(user_id, doc_id)
        try:
            r = self._sys.quiz.answer(user_id, doc_id, question_id, choice)
        except KeyError:
            raise ServiceError("question_not_found", f"Question {question_id!r} not found.")
        except ValueError as e:
            raise ServiceError("invalid_params", str(e))
        return AnswerOut(correct=r.correct, correct_index=r.correct_index, explanation=r.explanation, already_answered=r.already_answered)

    def feedback(self, user_id: str, doc_id: str, body: FeedbackBody) -> Ack:
        self._owned(user_id, doc_id)
        self._sys.tutor.feedback(user_id, doc_id, body.concept_ids, body.verdict)
        return Ack()

    def progress(self, user_id: str, doc_id: str, section_ordinal: int) -> Ack:
        """The reader finished a section: they have now been exposed to its concepts and are further along."""
        self._owned(user_id, doc_id)
        self._sys.tutor.finished_section(user_id, doc_id, section_ordinal)
        return Ack()

    @staticmethod
    def _q(q) -> QuizQuestionOut:
        return QuizQuestionOut(id=q.id, concept_id=q.concept_id, concept=q.concept, question=q.question, options=q.options,
                               difficulty=q.difficulty, kind=q.kind)
