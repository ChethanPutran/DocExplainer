"""Quizzes and the initial diagnostic.

One model call writes a whole quiz. Questions target concepts the reader has met and is least sure
about (where an answer teaches us most). The correct answer stays on the server until the reader answers,
each question counts once, and a graded answer is the strongest evidence the learner model gets.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..common.knowledge import Concept
from ..common.learner import BeliefChange, LearnerProfile
from ..common.ports import Embedder, JsonLLM, VectorIndex
from . import prompts
from .ports import KnowledgeReader, LearnerPort
from .quiz_store import SqliteQuizStore


@dataclass
class QuizQuestion:
    """What the reader is shown. (The answer is not here.)"""
    id: str
    concept_id: str
    concept: str
    question: str
    options: List[str]
    difficulty: float
    kind: str


@dataclass
class QuizResult:
    correct: bool
    correct_index: int
    explanation: str
    already_answered: bool = False
    changes: List[BeliefChange] = field(default_factory=list)


class QuizService:
    def __init__(self, llm: JsonLLM, embedder: Embedder, vectors: VectorIndex, knowledge: KnowledgeReader,
                 learner: LearnerPort, store: SqliteQuizStore, clock=time.time, seed: Optional[int] = None):
        self.llm, self.embedder, self.vectors, self.knowledge = llm, embedder, vectors, knowledge
        self.learner, self.store, self.clock, self._seed = learner, store, clock, seed

    # ------------------------------------------------------------------ practice quiz
    def practice(self, user_id: str, doc_id: str, position: int, n: int = 5) -> List[QuizQuestion]:
        """Questions on concepts the reader has met, preferring those they are least sure about."""
        concepts = [c for c in self.knowledge.concepts_upto(doc_id, position) if c.id]
        beliefs = self.learner.beliefs(user_id, doc_id, [c.id for c in concepts])
        # the sweet spot is where a question is informative AND answerable: belief in the middle of the range
        ranked = sorted(concepts, key=lambda c: (abs(beliefs[c.id].p_known - 0.5), -c.importance, c.id))
        return self._make(user_id, doc_id, ranked[:n], position, "practice")

    # ------------------------------------------------------------------ diagnostic
    def start_diagnostic(self, user_id: str, doc_id: str, level: str, known_topics: Sequence[str] = (),
                         questions: int = 6, last_ordinal: int = 10 ** 9) -> tuple:
        """Record the reader's own account of where they stand, then ask a few well-chosen questions."""
        known_ids = [c.id for name in known_topics for c in self.knowledge.concepts_in_text(doc_id, name, last_ordinal)]
        profile = self.learner.start(user_id, doc_id, level, known_ids, {"known_topics": list(known_topics)})
        candidates = self.knowledge.diagnostic_candidates(doc_id)
        chosen = self.learner.plan_diagnostic(user_id, doc_id, [(c.id, c.importance) for c in candidates], questions,
                                              self.store.asked_concepts(user_id, doc_id, "diagnostic"))
        by_id = {c.id: c for c in candidates}
        return profile, self._make(user_id, doc_id, [by_id[i] for i in chosen if i in by_id], last_ordinal, "diagnostic")

    def finish_diagnostic(self, user_id: str, doc_id: str) -> LearnerProfile:
        """Turn the answers so far into the reader's starting point."""
        return self.learner.recalibrate(user_id, doc_id)

    # ------------------------------------------------------------------ answering
    def answer(self, user_id: str, doc_id: str, question_id: str, choice: int) -> QuizResult:
        item = self.store.get(question_id, user_id, doc_id)
        if item is None:
            raise KeyError(question_id)
        payload = item["payload"]
        if not 0 <= choice < len(payload["options"]):                       # invalid input is invalid whatever the state
            raise ValueError(f"choice must be between 0 and {len(payload['options']) - 1}")
        if item["answered"]:
            return QuizResult(bool(item["correct"]), payload["answer_index"], payload["explanation"], already_answered=True)
        correct = choice == payload["answer_index"]
        if not self.store.mark_answered(question_id, choice, correct):          # lost a race: someone answered first
            return QuizResult(correct, payload["answer_index"], payload["explanation"], already_answered=True)
        changes = self.learner.quiz_answer(user_id, doc_id, item["concept_id"], correct, payload["difficulty"])
        return QuizResult(correct, payload["answer_index"], payload["explanation"], changes=changes)

    # ------------------------------------------------------------------ writing the questions
    def _make(self, user_id: str, doc_id: str, concepts: Sequence[Concept], position: int, kind: str) -> List[QuizQuestion]:
        if not concepts:
            return []
        beliefs = self.learner.beliefs(user_id, doc_id, [c.id for c in concepts])
        glossary = self.knowledge.glossary(doc_id, position, [c.id for c in concepts])
        items = []
        for c in concepts:
            p = beliefs[c.id].p_known
            hits = self.vectors.search(doc_id, self.embedder.embed([c.name])[0], k=1, max_ordinal=position)
            items.append({"concept_id": c.id, "name": c.name, "definition": glossary.get(c.id) or c.definition,
                          "passage": hits[0].chunk.text if hits else "",
                          "difficulty": round(min(max(0.5 + (p - 0.5) * 0.7, 0.1), 0.9), 2)})
        system, user = prompts.quiz_prompt(items)
        raw = self.llm.ask_json(system, user)
        by_concept = {it["concept_id"]: it for it in items}
        out: List[QuizQuestion] = []
        for q in (raw.get("questions") if isinstance(raw, dict) else None) or []:
            item = self._valid(q, by_concept)
            if item is None:
                continue
            options, answer_index = self._shuffled(item["options"], item["answer_index"], f"{doc_id}:{item['concept_id']}:{item['question']}")
            qid = uuid.uuid4().hex
            meta = by_concept[item["concept_id"]]
            self.store.add(qid, user_id, doc_id, item["concept_id"], kind,
                           {"question": item["question"], "options": options, "answer_index": answer_index,
                            "explanation": item["explanation"], "difficulty": meta["difficulty"]}, self.clock())
            out.append(QuizQuestion(qid, item["concept_id"], meta["name"], item["question"], options, meta["difficulty"], kind))
        if not out:
            raise ValueError("The model did not return any usable quiz questions.")
        return out

    @staticmethod
    def _valid(q, by_concept) -> Optional[dict]:
        if not isinstance(q, dict) or q.get("concept_id") not in by_concept:
            return None
        options = q.get("options")
        question = q.get("question")
        index = q.get("answer_index")
        if not isinstance(question, str) or not question.strip() or not isinstance(options, list):
            return None
        options = [str(o).strip() for o in options]
        if len(options) != 4 or len(set(o.lower() for o in options)) != 4 or any(not o for o in options):
            return None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4:
            return None
        return {"concept_id": q["concept_id"], "question": question.strip(), "options": options, "answer_index": index,
                "explanation": str(q.get("explanation") or "").strip()}

    def _shuffled(self, options: List[str], answer_index: int, key: str):
        """Models like putting the right answer first. Shuffle (reproducibly under a fixed seed) and track where it went."""
        rng = random.Random(f"{self._seed}:{key}") if self._seed is not None else random.Random()
        order = list(range(4))
        rng.shuffle(order)
        return [options[i] for i in order], order.index(answer_index)
