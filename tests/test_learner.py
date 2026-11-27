"""The learner model: BKT maths, evidence rules, propagation over the concept graph, forgetting, diagnostics."""
import json
import random

import pytest

from doc_explainer.core.common.learner import Belief, Evidence, EvidenceKind, level_for
from doc_explainer.core.learner import (BKTParams, Candidate, LEVEL_PRIORS, LearnerTracker, SqliteLearnerStore,
                                        classify_question, decay, exposure, observe, plan_diagnostic)
from doc_explainer.core.learner.bkt import error_rates

DOC = "book"


class Graph:
    """A -> B -> C (A is a prerequisite of B, B of C), D unrelated."""
    def __init__(self, edges=(("A", "B"), ("B", "C"))):
        self.pre, self.dep = {}, {}
        for a, b in edges:
            self.pre.setdefault(b, []).append((a, 1.0))
            self.dep.setdefault(a, []).append((b, 1.0))

    def prerequisites_of(self, doc_id, concept_id):
        return self.pre.get(concept_id, [])

    def dependents_of(self, doc_id, concept_id):
        return self.dep.get(concept_id, [])


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tracker(tmp_path, clock):
    return LearnerTracker(SqliteLearnerStore(tmp_path / "learner.db"), Graph(), clock=clock)


def quiz(concept, correct, difficulty=0.5):
    return Evidence(EvidenceKind.QUIZ_ANSWER, [concept], correct=correct, difficulty=difficulty)


def p(tracker, concept, user="u"):
    return tracker.belief(user, DOC, concept).p_known


# ----------------------------------------------------------------------------- BKT maths
GRID = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]


@pytest.mark.parametrize("prior", GRID)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_a_correct_answer_raises_belief_and_a_wrong_one_never_does(prior, difficulty):
    up, down = observe(prior, True, difficulty=difficulty), observe(prior, False, difficulty=difficulty)
    assert 0 < down <= prior <= up < 1                       # right never lowers, wrong never raises
    if prior < 0.98:
        assert up > prior
    if prior >= 0.3:
        assert down < prior                                  # at a very low belief a wrong answer holds it steady instead


def test_belief_stays_strictly_inside_zero_and_one_however_long_the_streak():
    p_ = 0.5
    for _ in range(200):
        p_ = observe(p_, True)
    assert 0.9 < p_ <= 0.99
    for _ in range(200):
        p_ = observe(p_, False)
    assert 0.01 <= p_ < 0.2        # even a student who is always wrong sits near the learning rate, not at zero (standard BKT)


def test_weight_blends_between_ignoring_and_fully_trusting_an_observation():
    none, half, full = (observe(0.4, True, weight=w) for w in (0.0, 0.5, 1.0))
    assert none == pytest.approx(0.4) and 0.4 < half < full


def test_getting_a_hard_question_right_is_stronger_evidence_than_an_easy_one():
    assert observe(0.4, True, difficulty=0.9) > observe(0.4, True, difficulty=0.1)
    assert observe(0.7, False, difficulty=0.1) < observe(0.7, False, difficulty=0.9)     # missing an easy one says more
    guess_easy, slip_easy = error_rates(0.0)
    guess_hard, slip_hard = error_rates(1.0)
    assert guess_easy > guess_hard and slip_easy < slip_hard


def test_forgetting_drifts_towards_the_prior_and_evidence_slows_it():
    assert decay(0.9, 0.3, 0, 0) == 0.9
    assert 0.3 < decay(0.9, 0.3, 30, 0) < 0.9
    assert decay(0.9, 0.3, 365, 0) < decay(0.9, 0.3, 30, 0)
    assert decay(0.9, 0.3, 60, 20) > decay(0.9, 0.3, 60, 0)                                 # well-practised fades slower
    assert decay(0.1, 0.3, 90, 0) > 0.1                                                      # a low belief drifts UP to the prior
    assert exposure(0.5) > 0.5 and exposure(0.5) - 0.5 < 0.03 and exposure(0.99) <= 0.99


@pytest.mark.parametrize("knows,lo,hi", [(True, 0.9, 1.0), (False, 0.0, 0.4)])   # non-knowers get lucky guesses; still far from knowers
def test_a_simulated_student_is_tracked_to_the_right_answer(knows, lo, hi):
    finals = []
    for seed in range(25):
        rng, belief = random.Random(seed), 0.35
        guess, slip = error_rates(0.5)
        for _ in range(25):
            correct = rng.random() < ((1 - slip) if knows else guess)
            belief = observe(belief, correct)
        finals.append(belief)
    assert lo <= sum(finals) / len(finals) <= hi


def test_a_student_who_learns_part_way_is_followed_upwards():
    rng, belief, trace = random.Random(7), 0.2, []
    guess, slip = error_rates(0.5)
    for i in range(40):
        knows = i >= 12                                  # learns the concept after 12 questions
        belief = observe(belief, rng.random() < ((1 - slip) if knows else guess))
        trace.append(belief)
    assert sum(trace[:10]) / 10 < 0.45 and trace[-1] > 0.85


# ----------------------------------------------------------------------------- onboarding
def test_the_starting_level_sets_the_prior_for_every_concept(tracker):
    for level, prior in LEVEL_PRIORS.items():
        profile = tracker.start("u" + level, DOC, level=level)
        assert profile.prior == prior and profile.level == level
        assert p(tracker, "anything", "u" + level) == prior and tracker.belief("u" + level, DOC, "x").evidence == 0
    with pytest.raises(ValueError):
        tracker.start("u", DOC, level="wizard")


def test_topics_the_learner_says_they_know_lift_those_concepts_and_what_they_rest_on(tracker):
    tracker.start("u", DOC, level="beginner", known_concept_ids=["B"])
    assert p(tracker, "B") >= 0.85 and p(tracker, "A") >= 0.70                              # B, and its prerequisite A
    assert p(tracker, "C") == LEVEL_PRIORS["beginner"] and p(tracker, "D") == LEVEL_PRIORS["beginner"]
    stored = tracker.profile("u", DOC)
    assert stored.onboarding["level"] == "beginner"
    assert [e["kind"] for e in tracker.store.events("u", DOC)] == ["onboarding"]


def test_a_learner_who_never_onboarded_gets_a_sensible_default(tracker):
    assert tracker.profile("nobody", DOC).level == "intermediate" and p(tracker, "A", "nobody") == LEVEL_PRIORS["intermediate"]


# ----------------------------------------------------------------------------- evidence -> beliefs
def test_a_correct_quiz_answer_raises_the_concept_and_weakly_its_prerequisites(tracker):
    tracker.start("u", DOC, level="beginner")
    prior = LEVEL_PRIORS["beginner"]
    changes = tracker.record("u", DOC, quiz("B", True))
    assert p(tracker, "B") > prior + 0.2
    assert prior < p(tracker, "A") < p(tracker, "B")                                         # prerequisite: up, but less
    assert p(tracker, "C") == prior and p(tracker, "D") == prior                             # dependents and strangers untouched
    assert {c.concept_id for c in changes} == {"B", "A"} and all(c.after > c.before for c in changes)


def test_a_wrong_answer_lowers_the_concept_and_weakly_what_is_built_on_it(tracker):
    tracker.start("u", DOC, level="advanced", known_concept_ids=["B", "C"])
    before = {c: p(tracker, c) for c in "ABCD"}
    tracker.record("u", DOC, quiz("B", False))
    assert p(tracker, "B") < before["B"] - 0.2
    assert before["C"] - 0.2 < p(tracker, "C") < before["C"]                                 # dependent: down, but less
    assert p(tracker, "A") == before["A"] and p(tracker, "D") == before["D"]


def test_propagation_never_moves_a_belief_the_wrong_way(tracker):
    tracker.start("u", DOC, level="beginner", known_concept_ids=["A"])
    high_a = p(tracker, "A")
    tracker.record("u", DOC, quiz("B", False))                                              # failing B must not LOWER prerequisite A
    assert p(tracker, "A") == high_a
    tracker.start("v", DOC, level="expert", known_concept_ids=["C"])
    high_c = p(tracker, "C")
    tracker.record("v", DOC, quiz("B", True))                                               # passing B must not RAISE dependent C
    assert p(tracker, "C") == high_c


def test_the_kind_of_question_asked_is_evidence(tracker):
    tracker.start("u", DOC, level="intermediate")
    prior = LEVEL_PRIORS["intermediate"]
    tracker.record("u", DOC, Evidence(EvidenceKind.QUESTION_ASKED, ["B"], question_type="definition"))
    asked_what_is_b = p(tracker, "B")
    assert asked_what_is_b < prior                                                           # "what is B?" -> probably new to them
    tracker.record("u", DOC, Evidence(EvidenceKind.QUESTION_ASKED, ["C"], question_type="deeper"))
    assert p(tracker, "C") > prior                                                           # "why does C ...?" -> they know C
    assert p(tracker, "B") > asked_what_is_b                                                 # ... and so, probably, what C rests on
    assert p(tracker, "D") == prior
    assert tracker.record("u", DOC, Evidence(EvidenceKind.QUESTION_ASKED, ["D"], question_type="")) == []   # unclear question: no signal


def test_feedback_that_an_explanation_was_too_easy_or_too_hard_is_evidence(tracker):
    tracker.start("u", DOC, level="intermediate")
    prior = LEVEL_PRIORS["intermediate"]
    tracker.record("u", DOC, Evidence(EvidenceKind.EXPLANATION_FEEDBACK, ["C"], verdict="too_easy"))
    tracker.record("u", DOC, Evidence(EvidenceKind.EXPLANATION_FEEDBACK, ["D"], verdict="too_hard"))
    assert p(tracker, "C") > prior and p(tracker, "D") < prior
    assert tracker.record("u", DOC, Evidence(EvidenceKind.EXPLANATION_FEEDBACK, ["A"], verdict="just_right")) == []


def test_reading_counts_for_a_little_and_is_not_counted_as_evidence(tracker):
    tracker.start("u", DOC, level="beginner")
    tracker.record("u", DOC, Evidence(EvidenceKind.READ_SECTION, ["A"]))
    b = tracker.belief("u", DOC, "A")
    assert LEVEL_PRIORS["beginner"] < b.p_known < LEVEL_PRIORS["beginner"] + 0.04 and b.evidence == 0


def test_a_self_report_moves_the_belief_towards_what_they_say(tracker):
    tracker.start("u", DOC, level="beginner")
    tracker.record("u", DOC, Evidence(EvidenceKind.SELF_REPORT, ["A"], value=0.95))
    assert p(tracker, "A") > 0.6
    tracker.record("u", DOC, Evidence(EvidenceKind.SELF_REPORT, ["A"], value=0.05))
    assert p(tracker, "A") < 0.4


def test_evidence_counts_and_the_audit_trail(tracker, clock):
    tracker.start("u", DOC)
    tracker.record("u", DOC, quiz("B", True))
    tracker.record("u", DOC, quiz("B", True))
    assert tracker.belief("u", DOC, "B").evidence == 2
    events = tracker.store.events("u", DOC)
    quiz_events = [e for e in events if e["kind"] == "quiz_answer" and e["concept_id"] == "B"]
    assert len(quiz_events) == 2 and quiz_events[1]["before"] == pytest.approx(quiz_events[0]["after"])
    assert quiz_events[0]["payload"]["correct"] is True and quiz_events[0]["after"] > quiz_events[0]["before"]


# ----------------------------------------------------------------------------- time, persistence, isolation
def test_forgetting_is_applied_when_beliefs_are_read_not_stored(tracker, clock):
    tracker.start("u", DOC, level="beginner")
    tracker.record("u", DOC, quiz("D", True))
    tracker.record("u", DOC, quiz("D", True))
    fresh = p(tracker, "D")
    clock.t += 120 * 86400
    faded = p(tracker, "D")
    assert LEVEL_PRIORS["beginner"] < faded < fresh
    assert tracker.store.get_beliefs("u", DOC)["D"].p_known == pytest.approx(fresh)          # the stored value is untouched


def test_state_survives_a_restart_and_learners_and_documents_are_isolated(tmp_path, clock):
    db = tmp_path / "l.db"
    one = LearnerTracker(SqliteLearnerStore(db), Graph(), clock=clock)
    one.start("alice", DOC, level="beginner")
    one.record("alice", DOC, quiz("D", True))
    two = LearnerTracker(SqliteLearnerStore(db), Graph(), clock=clock)                       # a fresh process
    assert two.belief("alice", DOC, "D").p_known > LEVEL_PRIORS["beginner"]
    assert two.belief("bob", DOC, "D").p_known == LEVEL_PRIORS["intermediate"]               # other learner: untouched default
    assert two.belief("alice", "other-doc", "D").evidence == 0                               # other document: nothing known
    two.forget("alice", DOC)
    assert two.belief("alice", DOC, "D").evidence == 0 and two.store.events("alice", DOC) == []


def test_concurrent_recording_loses_nothing(tracker):
    import threading
    tracker.start("u", DOC)
    errors = []

    def work(i):
        try:
            for _ in range(10):
                tracker.record("u", DOC, quiz(f"X{i}", True))
        except Exception as e:
            errors.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert errors == [] and all(tracker.belief("u", DOC, f"X{i}").evidence == 10 for i in range(4))


# ----------------------------------------------------------------------------- the starting point after a diagnostic
def answers(tracker, user, results, start="intermediate"):
    tracker.start(user, DOC, level=start)
    for i, ok in enumerate(results):
        tracker.record(user, DOC, quiz(f"Q{i}", ok))                      # each on a DIFFERENT concept, as a diagnostic is
    return tracker.recalibrate(user, DOC)


def test_after_a_diagnostic_the_prior_reflects_how_the_learner_actually_did(tracker):
    strong = answers(tracker, "strong", [True] * 4, start="beginner")      # says beginner, aces everything
    assert strong.level in ("advanced", "expert") and strong.prior > 0.6 and strong.level == level_for(strong.prior)
    weak = answers(tracker, "weak", [False] * 4, start="advanced")         # says advanced, misses everything
    assert weak.level == "beginner" and weak.prior < 0.3
    middling = answers(tracker, "mid", [True, False, True, False])
    assert middling.level == "intermediate"
    assert strong.prior > middling.prior > weak.prior
    assert p(tracker, "never-tested", "strong") == pytest.approx(strong.prior)             # untested concepts follow suit
    assert tracker.recalibrate("nobody-tested", DOC).prior == LEVEL_PRIORS["intermediate"]   # no graded answers: unchanged


def test_a_lucky_streak_on_few_questions_is_discounted(tracker):
    one = answers(tracker, "one", [True])
    ten = answers(tracker, "ten", [True] * 10)
    assert one.prior < ten.prior                                                            # more evidence, more conviction
    assert one.prior < 0.7                                                                  # one right answer could be a guess


def test_a_learners_own_account_still_counts_a_little(tracker):
    modest = answers(tracker, "modest", [True, True, False, True], start="beginner")
    confident = answers(tracker, "confident", [True, True, False, True], start="expert")
    assert confident.prior > modest.prior                                                   # same answers, different self-report


def test_propagated_updates_are_not_counted_as_extra_answers(tracker):
    tracker.start("u", DOC, level="intermediate")
    tracker.record("u", DOC, quiz("B", True))                                               # also nudges B's prerequisite A
    events = [e for e in tracker.store.events("u", DOC) if e["kind"] == "quiz_answer"]
    assert {e["concept_id"] for e in events} == {"B", "A"}                                  # two beliefs changed ...
    profile = tracker.recalibrate("u", DOC)                                                 # ... but only ONE graded answer
    expected = (1 * ((1 * max(0.0, (1.0 - 0.25) / 0.75) + 1) / 3) + LEVEL_PRIORS["intermediate"]) / 2
    assert profile.prior == pytest.approx(expected)


def test_the_reading_position_only_moves_forward(tracker):
    assert tracker.position("u", DOC) is None
    assert tracker.advance_position("u", DOC, 40) == 40
    assert tracker.advance_position("u", DOC, 10) == 40                                      # going back does not un-read
    assert tracker.advance_position("u", DOC, 90) == 90 and tracker.position("u", DOC) == 90
    assert tracker.position("other", DOC) is None


# ----------------------------------------------------------------------------- which concepts to test
def beliefs_of(**kw):
    return {k: Belief(v, 1) for k, v in kw.items()}


def test_the_diagnostic_prefers_important_concepts_we_are_unsure_about():
    cands = [Candidate("minor", 1), Candidate("central", 6), Candidate("middling", 3)]
    assert plan_diagnostic(DOC, cands, {}, Graph(()), k=3) == ["central", "middling", "minor"]
    # an important concept we are already sure about is less informative than a less important unsure one
    sure = beliefs_of(central=0.5 + 0.49)
    assert plan_diagnostic(DOC, cands, sure, Graph(()), k=1)[0] == "middling"


def test_the_diagnostic_skips_what_is_settled_or_already_asked():
    cands = [Candidate(c, 3) for c in "WXYZ"]
    beliefs = beliefs_of(W=0.97, X=0.02)                                                    # both settled by evidence
    assert plan_diagnostic(DOC, cands, beliefs, Graph(()), k=4, already_asked=["Y"]) == ["Z"]
    assert plan_diagnostic(DOC, [], {}, Graph(()), k=3) == []
    assert plan_diagnostic(DOC, cands[:2], {}, Graph(()), k=10) == ["W", "X"]                # fewer than k available


def test_the_diagnostic_does_not_waste_questions_on_directly_related_concepts():
    cands = [Candidate("A", 5), Candidate("B", 5), Candidate("D", 4)]                       # B depends directly on A; D is unrelated
    assert plan_diagnostic(DOC, cands, {}, Graph(), k=2) == ["A", "D"]                       # B is skipped for the unrelated D
    assert plan_diagnostic(DOC, cands, {}, Graph(), k=3) == ["A", "D", "B"]                  # but still asked if nothing else is left
    chain = [Candidate("A", 5), Candidate("B", 5), Candidate("C", 5)]                         # A->B->C: only DIRECT neighbours are skipped
    assert plan_diagnostic(DOC, chain, {}, Graph(), k=2) == ["A", "C"]


def test_the_diagnostic_adapts_as_answers_arrive(tracker):
    tracker.start("u", DOC, level="intermediate")
    cands = [Candidate("A", 5), Candidate("D", 5)]
    first = plan_diagnostic(DOC, cands, tracker.beliefs("u", DOC), tracker.graph, k=1)[0]
    for _ in range(3):
        tracker.record("u", DOC, quiz(first, True))
    second = plan_diagnostic(DOC, cands, tracker.beliefs("u", DOC, ["A", "D"]), tracker.graph, k=1)[0]
    assert first == "A" and second == "D"                                                    # A is settled: ask about D next


# ----------------------------------------------------------------------------- reading the questions
@pytest.mark.parametrize("text,expected", [
    ("What is ATP?", "definition"), ("Define entropy", "definition"), ("what does photosynthesis mean", "definition"),
    ("Why does the Calvin cycle need ATP?", "deeper"), ("How does ATP relate to the light reactions?", "deeper"),
    ("Compare mitosis and meiosis", "deeper"), ("What if the light is removed?", "deeper"), ("What is the difference between X and Y?", "deeper"),
    ("Tell me about sugar", ""), ("", ""),
])
def test_questions_are_classified_by_what_they_reveal(text, expected):
    assert classify_question(text) == expected
