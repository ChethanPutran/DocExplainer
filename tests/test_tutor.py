"""The tutor: pitching answers to the reader, staying within what they have read, learning from what they do."""
import pytest

from doc_explainer.core.common.explanation import Explanation
from doc_explainer.core.common.knowledge import Concept
from doc_explainer.core.common.learner import Belief, BeliefChange, LearnerProfile, level_for
from doc_explainer.core.common.ports import Chunk, Hit
from doc_explainer.core.common.structure import NodeKind
from doc_explainer.core.document import parse_document
from doc_explainer.core.tutor import (FocusConcept, QuizService, SectionConcepts, SqliteQuizStore, TutorService,
                                      plan_adaptation)
from doc_explainer.core.tutor.policy import GAP, KNOWN, MAX_NEW_TERMS

from fakes import HashingEmbedder

BOOK = """# Photosynthesis

Photosynthesis converts light energy into chemical energy in plants.

## Light reactions

Chlorophyll absorbs light and splits water. The light reactions produce ATP.

## Calvin cycle

The Calvin cycle uses ATP from the light reactions to fix carbon dioxide into sugar.

## Cellular respiration

Cellular respiration releases the energy stored in sugar.
"""


def concept(cid, name, node, structure, definition="", importance=2):
    return Concept(id=cid, name=name, introduced_ordinal=structure.node(node).ordinal, first_node_id=node,
                   definition=definition, importance=importance)


class Fixture:
    """A tiny book, its concepts, and fake neighbours for the tutor to talk to."""

    def __init__(self, tmp_path):
        (tmp_path / "b.md").write_text(BOOK, encoding="utf-8")
        self.structure = parse_document(tmp_path / "b.md", "book")
        s = self.structure
        self.concepts = {c.id: c for c in [
            concept("c1", "photosynthesis", "s1/p1", s, "Making chemical energy from light.", 3),
            concept("c2", "chlorophyll", "s1.1/p1", s, "A green pigment.", 2),
            concept("c3", "ATP", "s1.1/p1", s, "The cell's energy currency.", 3),
            concept("c4", "Calvin cycle", "s1.2/p1", s, "The sugar-making stage.", 3),
            concept("c5", "sugar", "s1.2/p1", s, "", 2),
            concept("c6", "cellular respiration", "s1.3/p1", s, "Releasing energy from sugar.", 2)]}
        self.prereq = {"c4": ["c3"], "c6": ["c5"]}                    # ATP -> Calvin cycle ; sugar -> respiration
        self.beliefs = {}
        self.prior = 0.35
        self.calls = []
        self.position = None
        self.searches = []
        self.llm_calls = []
        self.llm_reply = {"explanation": "An explanation.", "known_concepts_used": ["chlorophyll"],
                          "unknown_concepts_explained": ["ATP"], "follow_up_questions": ["Why?", "How?", "What next?", "extra"]}
        self.chunks = [Chunk("s1/p1", "Photosynthesis converts light energy.", "s1/p1", "paragraph", s.node("s1/p1").ordinal, 1, "s1", "Photosynthesis"),
                       Chunk("s1.1/p1", "Chlorophyll absorbs light; the light reactions produce ATP.", "s1.1/p1", "paragraph", s.node("s1.1/p1").ordinal, 1, "s1.1", "Photosynthesis > Light reactions"),
                       Chunk("s1.3/p1", "Cellular respiration releases energy.", "s1.3/p1", "paragraph", s.node("s1.3/p1").ordinal, 1, "s1.3", "Photosynthesis > Cellular respiration")]

    # ---- KnowledgeReader
    def section_concepts(self, doc_id, structure, section_id):
        sec = structure.node(section_id)
        lo, hi = sec.ordinal, structure.own_end(section_id)
        return SectionConcepts([c for c in self.concepts.values() if lo <= c.introduced_ordinal <= hi],
                               [c for c in self.concepts.values() if c.introduced_ordinal < lo and c.id in ("c3",) and section_id == "s1.2"])

    def concepts_in_text(self, doc_id, text, upto):
        t = text.lower()
        return [c for c in self.concepts.values() if c.name.lower() in t and c.introduced_ordinal <= upto]

    def concept(self, doc_id, cid):
        return self.concepts.get(cid)

    def concepts_upto(self, doc_id, ordinal):
        return [c for c in self.concepts.values() if c.introduced_ordinal <= ordinal]

    def prerequisites(self, doc_id, cid, depth, upto):
        return [(self.concepts[p], 1) for p in self.prereq.get(cid, []) if self.concepts[p].introduced_ordinal <= upto]

    def glossary(self, doc_id, ordinal, ids):
        return {i: self.concepts[i].definition for i in ids if self.concepts[i].definition and self.concepts[i].introduced_ordinal <= ordinal}

    def diagnostic_candidates(self, doc_id):
        return sorted(self.concepts.values(), key=lambda c: -c.importance)

    # ---- LearnerPort
    def get_beliefs(self, user_id, doc_id, ids):
        return {i: self.beliefs.get((user_id, i), Belief(self.prior, 0)) for i in ids}

    def profile(self, user_id, doc_id):
        return LearnerProfile(user_id, doc_id, self.prior, level_for(self.prior))

    def start(self, user_id, doc_id, level, known_ids, onboarding):
        self.calls.append(("start", user_id, level, list(known_ids), onboarding))
        return LearnerProfile(user_id, doc_id, 0.5, level, onboarding)

    def question_asked(self, user_id, doc_id, text, ids):
        self.calls.append(("question_asked", user_id, text, list(ids)))

    def explanation_feedback(self, user_id, doc_id, ids, verdict):
        self.calls.append(("feedback", user_id, list(ids), verdict))

    def read_section(self, user_id, doc_id, ids):
        self.calls.append(("read_section", user_id, list(ids)))

    def quiz_answer(self, user_id, doc_id, cid, correct, difficulty):
        self.calls.append(("quiz_answer", user_id, cid, correct, difficulty))
        return [BeliefChange(cid, 0.4, 0.6 if correct else 0.3)]

    def plan_diagnostic(self, user_id, doc_id, candidates, k, asked):
        self.calls.append(("plan_diagnostic", k, list(asked)))
        return [cid for cid, _ in candidates if cid not in asked][:k]

    def recalibrate(self, user_id, doc_id):
        self.calls.append(("recalibrate", user_id))
        return LearnerProfile(user_id, doc_id, 0.6, "advanced")

    def advance_position(self, user_id, doc_id, ordinal):
        self.calls.append(("advance", ordinal))
        self.position = max(ordinal, self.position or 0)
        return self.position

    # ---- StructureSource / VectorIndex / LLM
    def load(self, doc_id):
        return self.structure

    def search(self, doc_id, vector, k=5, max_ordinal=None):
        self.searches.append(max_ordinal)
        return [Hit(c, 0.9) for c in self.chunks if max_ordinal is None or c.ordinal <= max_ordinal][:k]

    def ask_json(self, system, user):
        self.llm_calls.append((system, user))
        return self.llm_reply(user) if callable(self.llm_reply) else self.llm_reply

    def stream_text(self, system, user):
        """The tutor's reply as streamed text. ``llm_reply`` keeps its dict shape: explanation + follow_up_questions."""
        self.llm_calls.append((system, user))
        reply = self.llm_reply(user) if callable(self.llm_reply) else self.llm_reply
        text = reply.get("explanation") if isinstance(reply, dict) else reply
        text = text if isinstance(text, str) else ""
        questions = reply.get("follow_up_questions") if isinstance(reply, dict) else None
        if isinstance(questions, list):
            text += "\n===FOLLOW-UPS===\n" + "\n".join(f"- {q}" for q in questions)
        for i in range(0, len(text), 5):
            yield text[i:i + 5]


class Knowledge:
    def __init__(self, f): self.f = f
    def __getattr__(self, name): return getattr(self.f, name)


class Learner:
    """LearnerPort view of the fixture; `beliefs` and `position` map to the fixture's names."""
    def __init__(self, f): self.f = f
    def beliefs(self, user_id, doc_id, ids): return self.f.get_beliefs(user_id, doc_id, ids)
    def position(self, user_id, doc_id): return self.f.position
    def __getattr__(self, name): return getattr(self.f, name)


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


@pytest.fixture
def tutor(fx):
    class Vectors:
        def search(_, *a, **k): return fx.search(*a, **k)
    return TutorService(fx, HashingEmbedder(), Vectors(), Knowledge(fx), Learner(fx), fx)


# ----------------------------------------------------------------------------- the adaptation policy
def focus(concepts, ps, evidence=3):
    return [FocusConcept(c, p, evidence) for c, p in zip(concepts, ps)]


def many(n, importance=2):
    return [Concept(id=f"k{i}", name=f"term{i}", introduced_ordinal=i, importance=importance) for i in range(n)]


def test_what_is_known_gets_assumed_what_is_missing_gets_defined_in_between_is_refreshed():
    cs = many(4)
    plan = plan_adaptation(focus(cs, [0.9, 0.55, 0.2, 0.05]))
    assert [c.name for c in plan.assume] == ["term0"]
    assert [c.name for c in plan.refresh] == ["term1"]
    assert {c.name for c in plan.define} == {"term2", "term3"} == {c.name for c in plan.gaps}
    assert KNOWN == 0.70 and GAP == 0.40


@pytest.mark.parametrize("p,level", [(0.1, "beginner"), (0.4, "intermediate"), (0.65, "advanced"), (0.9, "expert")])
def test_the_level_follows_the_importance_weighted_mastery(p, level):
    assert plan_adaptation(focus(many(3), [p] * 3)).level == level
    heavy = [Concept("a", "central", importance=9), Concept("b", "minor", importance=1)]
    assert plan_adaptation(focus(heavy, [0.9, 0.1])).level in ("advanced", "expert")        # the central concept dominates
    assert plan_adaptation(focus(heavy, [0.1, 0.9])).level in ("beginner", "intermediate")


def test_a_beginner_is_never_given_more_new_terms_than_they_can_absorb():
    gaps = many(8)
    for p, level in ((0.05, "beginner"), (0.35, "intermediate")):
        plan = plan_adaptation(focus(gaps, [p] * 8))
        assert plan.level == level and len(plan.define) == MAX_NEW_TERMS[level] and len(plan.gaps) == 8
    assert MAX_NEW_TERMS["beginner"] < MAX_NEW_TERMS["expert"]


def test_the_most_important_and_least_known_are_defined_first():
    cs = [Concept("a", "minor", importance=1, introduced_ordinal=1), Concept("b", "central", importance=5, introduced_ordinal=2),
          Concept("c", "medium", importance=3, introduced_ordinal=3)]
    plan = plan_adaptation(focus(cs, [0.1, 0.3, 0.1]))
    assert [c.name for c in plan.define][:2] == ["central", "medium"] or plan.define[0].name == "central"


def test_summaries_condense_and_define_nothing():
    cs = many(3)
    explain, summary = plan_adaptation(focus(cs, [0.1] * 3), "explain"), plan_adaptation(focus(cs, [0.1] * 3), "summarize")
    assert summary.define == [] and summary.max_new_terms == 0 and explain.define
    assert summary.words[1] < explain.words[1]


def test_with_nothing_to_go_on_the_default_prior_decides_and_the_depth_stays_adaptive():
    plan = plan_adaptation([], default_prior=0.8)
    assert plan.level == "expert" and plan.depth == "adaptive" and plan.define == [] and plan.assume == []
    assert plan_adaptation(focus(many(2), [0.5, 0.5], evidence=0)).depth == "adaptive"        # we have never tested them
    assert plan_adaptation(focus(many(2), [0.5, 0.5], evidence=5)).depth == "fixed"


# ----------------------------------------------------------------------------- the same question, two readers
def test_the_same_question_is_answered_differently_for_a_beginner_and_an_expert(tutor, fx):
    text = "The light reactions produce ATP, which the Calvin cycle needs."
    fx.beliefs = {("newbie", c): Belief(0.1, 2) for c in fx.concepts} | {("expert", c): Belief(0.95, 6) for c in fx.concepts}
    tutor.explain("newbie", "book", text, fx.structure.node("s1.2").ordinal)
    tutor.explain("expert", "book", text, fx.structure.node("s1.2").ordinal)
    (sys_n, user_n), (sys_e, user_e) = fx.llm_calls
    assert "READER LEVEL: beginner" in sys_n and "READER LEVEL: expert" in sys_e
    assert "does NOT yet know" in sys_n and "ATP = The cell's energy currency." in sys_n      # defined, using the text's own words
    assert "does NOT yet know" not in sys_e and "already knows (do not explain)" in sys_e     # skipped for the expert
    assert "at most 2 new terms" in sys_n and "at most" not in sys_e
    assert "plain everyday words" in sys_n and "terse and rigorous" in sys_e
    assert "no jargon" not in sys_e


def test_the_answer_reports_how_it_was_pitched(tutor, fx):
    fx.beliefs = {("u", c): Belief(0.1, 2) for c in fx.concepts}
    answer = tutor.explain("u", "book", "The light reactions produce ATP.", fx.structure.node("s1.1").ordinal)
    e = answer.explanation
    assert isinstance(e, Explanation) and answer.text == "An explanation."
    assert e.style.level.value == "beginner" and e.style.depth.value in ("adaptive", "fixed")
    assert e.context_used["section"] == "Photosynthesis > Light reactions" and e.context_used["mode"] == "explain"
    assert e.metadata.gap_bridging and e.metadata.estimated_complexity > 0.8
    assert len(e.follow_up_questions) == 3                                                    # capped at three
    assert {s["section"] for s in e.context_used["sources"]} <= {"Photosynthesis", "Photosynthesis > Light reactions"}


# ----------------------------------------------------------------------------- never beyond what has been read
def test_retrieval_is_limited_to_the_readers_position_so_nothing_later_can_leak(tutor, fx):
    s = fx.structure
    answer = tutor.explain("u", "book", "ATP", s.node("s1.1").ordinal)
    assert fx.searches == [s.own_end("s1.1")] and answer.position == s.own_end("s1.1")
    sent = fx.llm_calls[0][1]
    assert "Chlorophyll absorbs light; the light reactions produce ATP." in sent
    assert "Cellular respiration releases energy." not in sent                               # a later section never reaches the model


def test_only_earlier_sections_are_suggested_as_resources(tutor, fx):
    fx.beliefs = {("u", c): Belief(0.05, 2) for c in fx.concepts}
    s = fx.structure
    answer = tutor.explain("u", "book", "The Calvin cycle uses ATP.", s.node("s1.2").ordinal)
    res = [r for r in answer.explanation.resources if r.type.value == "section"]            # (web links are tested below)
    assert any(r.url == "doc://book#s1.1" and "ATP" in r.description and r.source == "This document" for r in res)   # taught in "Light reactions"
    section_ids = {r.url.split("#")[1] for r in res}
    assert all(s.node(i).ordinal < s.node("s1.2").ordinal for i in section_ids)               # nothing at or after the current section
    assert {sg.concept for sg in answer.explanation.suggested_resources} >= {"ATP"}


def test_a_reader_who_knows_everything_gets_no_remedial_pointers(tutor, fx):
    fx.beliefs = {("u", c): Belief(0.95, 5) for c in fx.concepts}
    answer = tutor.explain("u", "book", "The Calvin cycle uses ATP.", fx.structure.node("s1.2").ordinal)
    assert answer.explanation.resources == [] and answer.explanation.suggested_resources == []
    assert answer.plan.define == []


def test_internet_resources_are_search_links_for_the_concepts_the_reader_lacks(tutor, fx):
    fx.beliefs = {("u", c): Belief(0.05, 2) for c in fx.concepts}
    res = tutor.explain("u", "book", "The Calvin cycle uses ATP.", fx.structure.node("s1.2").ordinal).explanation.resources
    web = [r for r in res if r.type.value != "section"]
    assert {r.source for r in web} >= {"YouTube", "Wikipedia"}
    assert all(r.url.startswith("https://") for r in web)
    video = next(r for r in web if r.source == "YouTube")
    assert "youtube.com/results?search_query=" in video.url and "lecture" in video.url


# ----------------------------------------------------------------------------- learning from what they do
def test_a_question_is_recorded_as_evidence_before_the_answer_is_pitched(tutor, fx):
    order = []
    fx.question_asked = lambda user_id, doc_id, text, ids: order.append("recorded") or fx.calls.append(("question_asked", user_id, text, list(ids)))
    original = fx.stream_text
    fx.stream_text = lambda s, u: order.append("llm") or original(s, u)
    tutor.ask("u", "book", "What is ATP?", fx.structure.node("s1.1").ordinal)
    assert order == ["recorded", "llm"]
    call = next(c for c in fx.calls if c[0] == "question_asked")
    assert call[1] == "u" and call[2] == "What is ATP?" and "c3" in call[3]


def test_feedback_and_finishing_a_section_flow_into_the_learner_model(tutor, fx):
    tutor.feedback("u", "book", ["c3", "c2"], "too_hard")
    assert ("feedback", "u", ["c3", "c2"], "too_hard") in fx.calls
    tutor.finished_section("u", "book", fx.structure.node("s1.1").ordinal)
    read = next(c for c in fx.calls if c[0] == "read_section")
    assert set(read[2]) == {"c2", "c3"}                                                       # what that section introduced
    assert ("advance", fx.structure.own_end("s1.1")) in fx.calls


def test_summaries_need_no_retrieval_and_define_no_terms(tutor, fx):
    fx.beliefs = {("u", c): Belief(0.05, 2) for c in fx.concepts}
    tutor.summarize("u", "book", "The Calvin cycle uses ATP to fix carbon dioxide.", fx.structure.node("s1.2").ordinal)
    assert fx.searches == []
    system = fx.llm_calls[0][0]
    assert "Summarise; do not teach new material." in system and "does NOT yet know" not in system


# ----------------------------------------------------------------------------- where the reader is
def test_the_position_comes_from_the_section_given_else_the_stored_one_else_the_start(tutor, fx):
    s = fx.structure
    assert tutor.locate(s, s.node("s1.2").ordinal)[0].id == "s1.2"
    assert tutor.locate(s, s.node("s1.2/p1").ordinal)[0].id == "s1.2"                         # a paragraph ordinal -> its section
    assert tutor.locate(s, s.node("s1.2/p1/n1").ordinal)[0].id == "s1.2"
    assert tutor.locate(s, None)[0].id == "s1"                                                # nothing known: the start
    fx.position = s.own_end("s1.3")
    assert tutor.locate(s, None, "u")[0].id == "s1.3"                                         # the learner's stored position
    assert tutor.locate(s, 99999)[0].id == "s1"                                                # nonsense: safe default, no crash
    section, position = tutor.locate(s, s.node("s1.1").ordinal)
    assert position == s.own_end("s1.1") > section.ordinal


def test_a_reply_without_an_explanation_is_an_error_not_an_empty_answer(tutor, fx):
    for bad in ({}, {"explanation": ""}, {"explanation": 5}, "", None):
        fx.llm_reply = bad
        with pytest.raises(ValueError):
            tutor.explain("u", "book", "ATP", fx.structure.node("s1.1").ordinal)


def test_odd_optional_fields_from_the_model_are_tolerated(tutor, fx):
    fx.llm_reply = {"explanation": "Fine.", "known_concepts_used": "not a list", "follow_up_questions": {"a": 1}}
    a = tutor.explain("u", "book", "ATP", fx.structure.node("s1.1").ordinal)
    assert a.text == "Fine." and a.explanation.follow_up_questions == []


# ----------------------------------------------------------------------------- quizzes
@pytest.fixture
def quiz(fx, tmp_path):
    class Vectors:
        def search(_, *a, **k): return fx.search(*a, **k)

    def questions(user):
        ids = [l.split()[1].rstrip(":") for l in user.splitlines() if l.startswith("CONCEPT ")]
        return {"questions": [{"concept_id": i, "question": f"Question about {i}?", "options": ["right", "wrong 1", "wrong 2", "wrong 3"],
                               "answer_index": 0, "explanation": f"Because {i}."} for i in ids]}
    fx.llm_reply = questions
    return QuizService(fx, HashingEmbedder(), Vectors(), Knowledge(fx), Learner(fx), SqliteQuizStore(tmp_path / "q.db"), seed=1)


def test_a_practice_quiz_covers_concepts_already_met_preferring_the_uncertain(quiz, fx):
    s = fx.structure
    fx.beliefs = {("u", "c1"): Belief(0.95, 4), ("u", "c2"): Belief(0.5, 1), ("u", "c3"): Belief(0.1, 2)}
    qs = quiz.practice("u", "book", s.own_end("s1.1"), n=2)
    assert len(fx.llm_calls) == 1                                                             # one model call for the whole quiz
    assert [q.concept_id for q in qs] == ["c2", "c3"] or {q.concept_id for q in qs} == {"c2", "c3"}   # not the settled c1, nor any later concept
    assert all(fx.concepts[q.concept_id].introduced_ordinal <= s.own_end("s1.1") for q in qs)
    assert all(len(q.options) == 4 and q.kind == "practice" for q in qs) and not hasattr(qs[0], "answer_index")


def test_harder_questions_for_those_who_know_more(quiz, fx):
    fx.beliefs = {("u", "c2"): Belief(0.9, 3), ("u", "c3"): Belief(0.1, 3)}
    qs = {q.concept_id: q for q in quiz.practice("u", "book", fx.structure.own_end("s1.1"), n=6)}
    assert qs["c2"].difficulty > qs["c3"].difficulty and all(0.1 <= q.difficulty <= 0.9 for q in qs.values())


def test_the_right_answer_is_not_always_first_and_is_tracked_through_the_shuffle(quiz, fx):
    qs = quiz.practice("u", "book", 10 ** 6, n=6)
    chosen = []
    for q in qs:
        stored = quiz.store.get(q.id, "u", "book")["payload"]
        assert q.options[stored["answer_index"]] == "right"                                   # wherever it moved to
        chosen.append(stored["answer_index"])
    assert len(set(chosen)) > 1


def test_answers_are_graded_server_side_and_feed_the_learner_model_once(quiz, fx):
    q = quiz.practice("u", "book", 10 ** 6, n=1)[0]
    stored = quiz.store.get(q.id, "u", "book")["payload"]
    wrong = (stored["answer_index"] + 1) % 4
    res = quiz.answer("u", "book", q.id, wrong)
    assert res.correct is False and res.correct_index == stored["answer_index"] and res.explanation.startswith("Because")
    assert fx.calls[-1][:4] == ("quiz_answer", "u", q.concept_id, False) and res.changes
    again = quiz.answer("u", "book", q.id, stored["answer_index"])                            # a second try cannot change the record
    assert again.already_answered and again.correct is False
    assert len([c for c in fx.calls if c[0] == "quiz_answer"]) == 1


def test_a_correct_answer_and_the_rules_around_answering(quiz, fx):
    q = quiz.practice("u", "book", 10 ** 6, n=1)[0]
    right = quiz.store.get(q.id, "u", "book")["payload"]["answer_index"]
    assert quiz.answer("u", "book", q.id, right).correct is True
    with pytest.raises(KeyError):
        quiz.answer("mallory", "book", q.id, right)                                           # someone else's question
    with pytest.raises(KeyError):
        quiz.answer("u", "other-doc", q.id, right)
    with pytest.raises(KeyError):
        quiz.answer("u", "book", "no-such-question", 0)
    q2 = quiz.practice("u", "book", 10 ** 6, n=1)[0]
    with pytest.raises(ValueError):
        quiz.answer("u", "book", q2.id, 7)


def test_unusable_questions_from_the_model_are_dropped(quiz, fx):
    good = {"concept_id": "c2", "question": "ok?", "options": ["a", "b", "c", "d"], "answer_index": 1, "explanation": "x"}
    fx.llm_reply = {"questions": [
        good, {**good, "concept_id": "zzz"}, {**good, "options": ["a", "a", "c", "d"]}, {**good, "options": ["a", "b", "c"]},
        {**good, "answer_index": 4}, {**good, "answer_index": True}, {**good, "question": ""}, "junk", {**good, "options": ["a", "b", "c", ""]}]}
    fx.beliefs = {("u", "c2"): Belief(0.5, 1)}                  # c2 is the most informative concept, so it is the one asked about
    qs = quiz.practice("u", "book", 10 ** 6, n=3)
    assert len(qs) == 1 and qs[0].concept_id == "c2"
    fx.llm_reply = {"questions": [{**good, "options": ["a", "a", "a", "a"]}]}
    with pytest.raises(ValueError, match="usable quiz"):
        quiz.practice("u", "book", 10 ** 6, n=2)
    assert quiz.practice("u", "book", -1, n=3) == []                                          # nothing read yet: nothing to ask


# ----------------------------------------------------------------------------- the diagnostic
def test_the_diagnostic_records_the_learners_own_account_then_asks_well_chosen_questions(quiz, fx):
    profile, qs = quiz.start_diagnostic("u", "book", "beginner", known_topics=["ATP", "photosynthesis"], questions=3)
    start = next(c for c in fx.calls if c[0] == "start")
    assert start[2] == "beginner" and set(start[3]) == {"c3", "c1"} and start[4]["known_topics"] == ["ATP", "photosynthesis"]
    assert len(qs) == 3 and {q.kind for q in qs} == {"diagnostic"} and profile.level == "beginner"
    plan = next(c for c in fx.calls if c[0] == "plan_diagnostic")
    assert plan[1] == 3 and plan[2] == []
    _, more = quiz.start_diagnostic("u", "book", "beginner", questions=3)
    asked_before = {q.concept_id for q in qs}
    assert asked_before.isdisjoint({q.concept_id for q in more})                              # never re-asks a concept
    assert quiz.finish_diagnostic("u", "book").level == "advanced" and ("recalibrate", "u") in fx.calls
