"""TutorService: answers a reader's questions at their position in the document, pitched to what they know.

For every request it works out where the reader is (a reading-order position), which concepts are in play,
what the reader is believed to know about each, and from that an adaptation plan (policy.py). Retrieval is
limited to what has been read so far. What the reader asks, and how they react, flows back into the learner
model, so the next answer is pitched a little better.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..common.dataclasses import ExplanationStyle
from ..common.enums import ExplanationDepth, ExplanationLevel
from ..common.explanation import Explanation, ExplanationMetadata
from ..common.knowledge import Concept
from ..common.ports import Embedder, Hit, TextLLM, VectorIndex
from ..common.structure import DocumentStructure, Node, NodeKind
from . import prompts
from .policy import AdaptationPlan, FocusConcept, plan_adaptation, with_level
from .ports import KnowledgeReader, LearnerPort, StructureSource
from .resources import in_document_resources
from .splitter import FollowUpSplitter
from .web_resources import web_resources

logger = logging.getLogger(__name__)

PASSAGES = 4
MAX_FOCUS = 8


@dataclass
class Answer:
    text: str
    explanation: Explanation
    plan: AdaptationPlan
    position: int
    section_id: str
    concept_ids: List[str] = field(default_factory=list)


class TutorService:
    def __init__(self, llm: TextLLM, embedder: Embedder, vectors: VectorIndex, knowledge: KnowledgeReader,
                 learner: LearnerPort, structures: StructureSource, clock=time.time):
        self.llm, self.embedder, self.vectors = llm, embedder, vectors
        self.knowledge, self.learner, self.structures, self.clock = knowledge, learner, structures, clock

    # ------------------------------------------------------------------ where is the reader?
    def locate(self, structure: DocumentStructure, section_ordinal: Optional[int], user_id: str = "") -> tuple:
        """(section node, position). The position is the end of the section's own content: all they have read."""
        node: Optional[Node] = None
        if section_ordinal is not None:
            node = next((n for n in structure.nodes if n.ordinal == section_ordinal), None)
        if node is not None and node.kind is not NodeKind.SECTION:
            node = structure.section_of(node.id)
        if node is None:
            sections = structure.sections()
            known = self.learner.position(user_id, structure.doc_id) if user_id else None
            node = (structure.section_of(next(n.id for n in structure.nodes if n.ordinal == known))
                    if known is not None and any(n.ordinal == known for n in structure.nodes) else None) \
                or (sections[0] if sections else structure.root)
        position = structure.own_end(node.id) if node.kind is NodeKind.SECTION else node.ordinal
        return node, position

    # ------------------------------------------------------------------ the three things a reader asks for
    # Each is a stream of events: ("status", stage), ("delta", text), ("done", Answer). The plain calls below just
    # read the stream to the end, so streaming and non-streaming answers are the same thing.
    def stream(self, kind: str, user_id: str, doc_id: str, text: str, section_ordinal: Optional[int] = None,
               level: Optional[str] = None) -> Iterator[Tuple[str, object]]:
        """kind: explain | summarize | ask."""
        if kind not in ("explain", "summarize", "ask"):
            raise ValueError(f"Unknown request kind {kind!r}")
        yield ("status", "understanding")
        structure, section, position = self._where(user_id, doc_id, section_ordinal)
        focus = self._focus(user_id, doc_id, structure, section, position, text)
        if kind == "ask":   # what the reader asks says what they know: record it BEFORE pitching the answer
            self.learner.question_asked(user_id, doc_id, text, [f.id for f in focus[:3]])
        plan = self._plan(user_id, doc_id, focus, kind, level)
        hits: List[Hit] = []
        path = self._path(structure, section)
        if kind == "summarize":
            system, user = prompts.summarize_prompt(plan, {}, path, text)
        else:
            yield ("status", "searching")
            glossary = self.knowledge.glossary(doc_id, position, [c.id for c in plan.define])
            hits = self._retrieve(doc_id, text, position)
            if kind == "explain":
                system, user = prompts.explain_prompt(plan, glossary, path, text, hits,
                                                      structure.text_of(section.id)[:2500] if section.kind is NodeKind.SECTION else "")
            else:
                system, user = prompts.ask_prompt(plan, glossary, path, text, hits)
        yield ("status", "thinking")
        started, splitter, writing = self.clock(), FollowUpSplitter(prompts.FOLLOW_UP_MARK), False
        for piece in self.llm.stream_text(system, user):
            for out in splitter.feed(piece):
                if not writing:
                    writing = True
                    yield ("status", "writing")
                yield ("delta", out)
        for out in splitter.finish_text():
            if not writing:
                writing = True
                yield ("status", "writing")
            yield ("delta", out)
        body, follow_ups = splitter.result()
        if not body.strip():
            raise ValueError("The model's reply had no text.")
        yield ("status", "finishing")
        yield ("done", self._answer(doc_id, structure, section, position, plan, focus, hits, body, follow_ups, kind, started))

    def _collect(self, events: Iterable[Tuple[str, object]]) -> Answer:
        answer = None
        for kind, payload in events:
            if kind == "done":
                answer = payload
        return answer   # type: ignore[return-value]

    def explain(self, user_id: str, doc_id: str, selected_text: str, section_ordinal: Optional[int] = None,
                level: Optional[str] = None) -> Answer:
        return self._collect(self.stream("explain", user_id, doc_id, selected_text, section_ordinal, level))

    def summarize(self, user_id: str, doc_id: str, selected_text: str, section_ordinal: Optional[int] = None,
                  level: Optional[str] = None) -> Answer:
        return self._collect(self.stream("summarize", user_id, doc_id, selected_text, section_ordinal, level))

    def ask(self, user_id: str, doc_id: str, question: str, section_ordinal: Optional[int] = None,
            level: Optional[str] = None) -> Answer:
        return self._collect(self.stream("ask", user_id, doc_id, question, section_ordinal, level))

    def feedback(self, user_id: str, doc_id: str, concept_ids: Sequence[str], verdict: str) -> None:
        """The reader said an explanation was too_easy / too_hard / just_right."""
        self.learner.explanation_feedback(user_id, doc_id, list(concept_ids), verdict)

    def finished_section(self, user_id: str, doc_id: str, section_ordinal: int) -> None:
        """The reader got to the end of a section: record the exposure and move their position on."""
        structure = self.structures.load(doc_id)
        section, position = self.locate(structure, section_ordinal, user_id)
        if section.kind is NodeKind.SECTION:
            k = self.knowledge.section_concepts(doc_id, structure, section.id)
            self.learner.read_section(user_id, doc_id, [c.id for c in k.introduces])
        self.learner.advance_position(user_id, doc_id, position)

    # ------------------------------------------------------------------ internals
    def _where(self, user_id: str, doc_id: str, section_ordinal: Optional[int]):
        structure = self.structures.load(doc_id)
        section, position = self.locate(structure, section_ordinal, user_id)
        return structure, section, position

    def _path(self, structure: DocumentStructure, section: Node) -> str:
        return " > ".join(structure.section_path(section.id)) or structure.title

    def _plan(self, user_id, doc_id, focus, mode, level) -> AdaptationPlan:
        plan = plan_adaptation(self._with_beliefs(user_id, doc_id, focus), mode, self._prior(user_id, doc_id))
        return with_level(plan, level) if level else plan

    def _prior(self, user_id: str, doc_id: str) -> float:
        return self.learner.profile(user_id, doc_id).prior

    def _focus(self, user_id: str, doc_id: str, structure, section: Node, position: int, text: str) -> List[Concept]:
        """The concepts this request is about: those named in the text first, then the section's most important."""
        named = self.knowledge.concepts_in_text(doc_id, text, position)
        ranked = list(named)
        if section.kind is NodeKind.SECTION:
            k = self.knowledge.section_concepts(doc_id, structure, section.id)
            extra = sorted([*k.introduces, *k.uses_earlier], key=lambda c: -c.importance)
            ranked += [c for c in extra if c not in ranked][: max(0, 4 - len(named))]
        seen, out = set(), []
        for c in ranked:
            if c.id not in seen:
                seen.add(c.id)
                out.append(c)
        return out[:MAX_FOCUS]

    def _with_beliefs(self, user_id: str, doc_id: str, concepts: Sequence[Concept]) -> List[FocusConcept]:
        beliefs = self.learner.beliefs(user_id, doc_id, [c.id for c in concepts])
        return [FocusConcept(c, beliefs[c.id].p_known, beliefs[c.id].evidence) for c in concepts]

    def _retrieve(self, doc_id: str, query: str, position: int) -> List[Hit]:
        """Passages similar to the query, from what the reader has read and nothing later."""
        return self.vectors.search(doc_id, self.embedder.embed([query])[0], k=PASSAGES, max_ordinal=position)

    def _answer(self, doc_id, structure, section, position, plan, focus, hits, text, follow_ups, mode, started) -> Answer:
        gaps = plan.gaps
        prereqs: Dict[str, List[Concept]] = {}
        if section.kind is NodeKind.SECTION:
            for g in gaps[:3]:
                prereqs[g.id] = [c for c, _ in self.knowledge.prerequisites(doc_id, g.id, 1, position)]
            suggestions, resources = in_document_resources(doc_id, structure, gaps[:3], prereqs, section.id)
        else:
            suggestions, resources = [], []
        resources = resources + web_resources([g.name for g in gaps[:2]], ExplanationLevel(plan.level),
                                              context=structure.title)
        explanation = Explanation(
            explanation=text.strip(),
            style=ExplanationStyle(level=ExplanationLevel(plan.level), depth=ExplanationDepth(plan.depth),
                                   description=f"{plan.level} reader; mastery {plan.mastery:.0%} of the concepts in play."),
            context_used={"section": self._path(structure, section), "position": position, "mode": mode,
                          "concepts": [c.name for c in focus], "concept_ids": [c.id for c in focus],
                          "sources": [{"node": h.chunk.node_id, "page": h.chunk.page, "section": h.chunk.section_path} for h in hits]},
            known_concepts_used=[c.name for c in plan.assume[:5]],
            unknown_concepts_explained=[c.name for c in plan.define],
            follow_up_questions=follow_ups[:3],
            suggested_resources=suggestions,
            metadata=ExplanationMetadata(estimated_complexity=round(1 - plan.mastery, 3), user_knowledge_matched=bool(plan.assume),
                                         gap_bridging=bool(plan.define),
                                         processing_time_ms=round((self.clock() - started) * 1000, 1)),
            resources=resources)
        return Answer(text=explanation.explanation, explanation=explanation, plan=plan, position=position,
                      section_id=section.id, concept_ids=[c.id for c in focus])
