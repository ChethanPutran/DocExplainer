"""The whole system, wired by the orchestrator: real stores, real vector index, real adapters; only the LLM is scripted."""
import re
import threading
import time

import pytest

from doc_explainer.core.common.ports import LLMError
from doc_explainer.core.common.structure import EdgeKind
from doc_explainer.core.document import UnsupportedFormat
from doc_explainer.orchestrator import EmptyDocumentError, System, SystemConfig

from fakes import BOOK, SCRIPT, Brain, HashingEmbedder


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "book.md"
    p.write_text(BOOK, encoding="utf-8")
    return p


@pytest.fixture
def make(tmp_path):
    def build(brain=None, name="sys"):
        return System(SystemConfig(data_dir=tmp_path / name), llm=brain or Brain(), embedder=HashingEmbedder())
    return build


@pytest.fixture
def system(make, book):
    s = make()
    s.report = s.ingest(book, "book")
    return s


def ordinal(system, section_id):
    return system.structure("book").node(section_id).ordinal


# ----------------------------------------------------------------------------- ingestion
def test_ingesting_a_document_fills_every_store(system):
    r = system.report
    assert (r.title, r.format, r.sections, r.concepts, r.knowledge) == ("Photosynthesis", "markdown", 4, 8, "ready") and r.chunks >= 4
    assert system.vectors.count("book") == r.chunks
    assert system.structure_store.has("book") and system.content_store.exists("book")
    assert len(system.knowledge_store.concepts("book")) == 8 and system.knowledge_store.has_knowledge("book")
    cells = " ".join(system.structure_store.all_text_cells())
    assert "Photosynthesis converts light energy" not in cells                                 # the graph store holds no body text
    assert set(system.supported_extensions()) >= {".docx", ".html", ".md", ".pdf", ".pptx", ".txt"}


def test_progress_covers_all_three_stages_in_order(make, book):
    seen = []
    make().ingest(book, "book", progress=lambda stage, f: seen.append((stage, f)))
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and seen[0][1] == 0.0 and seen[-1] == ("done", 1.0)
    assert {s.split(":")[0] for s, _ in seen} >= {"reading the file", "embedding", "reading"}
    assert any(0.35 < f < 1.0 and s.startswith("reading:") for s, f in seen)                   # knowledge fills 35%..100%


def test_reingesting_does_not_redo_the_knowledge_work(make, book):
    brain = Brain()
    s = make(brain)
    s.ingest(book, "book")
    first = len(brain.calls)
    again = s.ingest(book, "book")
    assert len(brain.calls) == first and again.concepts == 8 and again.knowledge == "ready"     # resumed, nothing repeated
    assert s.vectors.count("book") == again.chunks                                              # but the index was refreshed, not doubled


def test_unreadable_and_unsupported_files_are_clear_errors(make, tmp_path):
    import pymupdf
    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open(); doc.new_page(); doc.save(str(blank)); doc.close()                      # a page with no text layer
    s = make()
    with pytest.raises(EmptyDocumentError, match="OCR"):
        s.ingest(blank, "scan")
    (tmp_path / "x.xyz").write_text("data")
    with pytest.raises(UnsupportedFormat):
        s.ingest(tmp_path / "x.xyz", "x")
    assert not s.structure_store.has("scan") and s.vectors.count("scan") == 0                   # nothing half-stored


def test_a_failing_model_does_not_destroy_an_otherwise_good_document(make, book):
    brain = Brain()
    brain.fail_knowledge = True
    s = make(brain)
    r = s.ingest(book, "book")
    assert r.knowledge == "failed" and "401" in r.knowledge_error and r.concepts == 0
    assert s.vectors.count("book") > 0 and s.structure_store.has("book")                          # still searchable
    brain.fail_knowledge = False
    answer = s.tutor.explain("u", "book", "The Calvin cycle fixes carbon dioxide.", ordinal(s, "s1.2"))
    assert answer.text.startswith("LEVEL=")                                                       # the tutor works, just less personalised
    assert s.quiz.practice("u", "book", 10 ** 6, n=3) == []                                       # no concepts to ask about
    recovered = s.ingest(book, "book")
    assert recovered.knowledge == "ready" and recovered.concepts == 8                             # a later run completes it


def test_removing_a_document_removes_everything_about_it(system):
    system.remove("book")
    assert system.vectors.count("book") == 0 and not system.structure_store.has("book") \
        and not system.content_store.exists("book") and system.knowledge_store.concepts("book") == []


# ----------------------------------------------------------------------------- where things are
def test_a_page_maps_to_the_section_that_holds_it(system):
    assert system.section_at("book", 1) == ordinal(system, "s1.3") or system.section_at("book", 1) in {ordinal(system, s) for s in ("s1", "s1.1", "s1.2", "s1.3")}
    s = system.structure("book")
    assert s.node(next(n.id for n in s.nodes if n.ordinal == system.section_at("book", 1))).kind.value == "section"
    assert system.section_at("book", 99) == system.section_at("book", 5)                          # past the end -> the last section


# ----------------------------------------------------------------------------- the tutor, end to end
def test_the_same_question_is_pitched_differently_for_a_newcomer_and_an_expert(system):
    system.tracker.start("newbie", "book", "beginner")
    system.tracker.start("expert", "book", "expert")
    text = "The Calvin cycle uses ATP from the light reactions."
    sec = ordinal(system, "s1.2")
    n, e = system.tutor.explain("newbie", "book", text, sec), system.tutor.explain("expert", "book", text, sec)
    assert n.text.startswith("LEVEL=beginner") and "ATP = The cell's energy currency." in n.text
    assert e.text == "LEVEL=expert; DEFINE=nothing"                                               # nothing to define for an expert
    assert n.explanation.style.level.value == "beginner" and e.explanation.style.level.value == "expert"
    assert any(r.url == "doc://book#s1.1" for r in n.explanation.resources)                       # the newcomer is sent back to "Light reactions"
    assert [r for r in e.explanation.resources if r.type.value == 'section'] == []


def test_answers_only_draw_on_what_the_reader_has_read(system):
    a = system.tutor.explain("u", "book", "ATP and the light reactions", ordinal(system, "s1.1"))
    sections = {s["section"] for s in a.explanation.context_used["sources"]}
    assert sections and all("Calvin" not in s and "respiration" not in s for s in sections)
    sent = system.llm.calls[-1][1]
    assert "Calvin cycle uses ATP" not in sent and "Cellular respiration releases" not in sent


def test_what_the_reader_asks_changes_what_we_believe_they_know(system):
    t = system.tracker
    t.start("u", "book", "intermediate")
    atp = system.knowledge_store.concepts("book")[0]
    atp = next(c for c in system.knowledge_store.concepts("book") if c.name == "ATP")
    before = t.belief("u", "book", atp.id).p_known
    system.tutor.ask("u", "book", "What is ATP?", ordinal(system, "s1.1"))
    after_definition = t.belief("u", "book", atp.id).p_known
    assert after_definition < before                                                              # "what is X?" -> probably new to them
    system.tutor.ask("u", "book", "Why does ATP matter for the light reactions?", ordinal(system, "s1.1"))
    assert t.belief("u", "book", atp.id).p_known > after_definition                               # a deeper question -> they know it
    kinds = [e["kind"] for e in system.learner_store.events("u", "book")]
    assert kinds.count("question_asked") >= 2


def test_feedback_and_progress_flow_into_the_model_and_the_position(system):
    atp = next(c for c in system.knowledge_store.concepts("book") if c.name == "ATP")
    system.tracker.start("u", "book", "intermediate")
    base = system.tracker.belief("u", "book", atp.id).p_known
    system.tutor.feedback("u", "book", [atp.id], "too_hard")
    assert system.tracker.belief("u", "book", atp.id).p_known < base
    system.tutor.finished_section("u", "book", ordinal(system, "s1.1"))
    assert system.tracker.position("u", "book") == system.structure("book").own_end("s1.1")
    system.tutor.finished_section("u", "book", ordinal(system, "s1"))
    assert system.tracker.position("u", "book") == system.structure("book").own_end("s1.1")      # never moves back


def test_a_diagnostic_changes_how_the_next_answer_is_pitched(system):
    text, sec = "The Calvin cycle uses ATP from the light reactions.", ordinal(system, "s1.2")
    profile, questions = system.quiz.start_diagnostic("learner", "book", "beginner", known_topics=["photosynthesis"], questions=4)
    assert len(questions) == 4 and all(q.kind == "diagnostic" for q in questions)
    before = system.tutor.explain("learner", "book", text, sec)
    assert before.text.startswith("LEVEL=beginner")
    for q in questions:                                                                           # answers every question correctly
        stored = system.quiz_store.get(q.id, "learner", "book")["payload"]
        assert system.quiz.answer("learner", "book", q.id, stored["answer_index"]).correct
    final = system.quiz.finish_diagnostic("learner", "book")
    assert final.prior > 0.6 and final.level in ("advanced", "expert")
    after = system.tutor.explain("learner", "book", text, sec)
    assert not after.text.startswith("LEVEL=beginner") and after.plan.mastery > before.plan.mastery
    assert len(after.plan.define) < len(before.plan.define) or after.plan.define == []             # fewer definitions now


def test_a_practice_quiz_updates_beliefs_through_the_whole_stack(system):
    system.tracker.start("u", "book", "beginner")
    position = system.structure("book").own_end("s1.2")
    qs = system.quiz.practice("u", "book", position, n=2)
    assert qs and all(system.knowledge_store.concept("book", q.concept_id).introduced_ordinal <= position for q in qs)
    q = qs[0]
    before = system.tracker.belief("u", "book", q.concept_id).p_known
    stored = system.quiz_store.get(q.id, "u", "book")["payload"]
    assert system.quiz.answer("u", "book", q.id, stored["answer_index"]).changes
    assert system.tracker.belief("u", "book", q.concept_id).p_known > before


def test_learners_do_not_affect_each_other(system):
    atp = next(c for c in system.knowledge_store.concepts("book") if c.name == "ATP")
    system.tracker.start("a", "book", "beginner")
    system.tracker.start("b", "book", "beginner")
    system.tutor.ask("a", "book", "What is ATP?", ordinal(system, "s1.1"))
    assert system.tracker.belief("b", "book", atp.id).p_known == system.tracker.profile("b", "book").prior


# ----------------------------------------------------------------------------- the structure is queryable
def test_the_structure_graph_and_the_knowledge_graph_agree_on_where_things_are(system):
    s = system.structure("book")
    for c in system.knowledge_store.concepts("book"):
        assert c.first_node_id in s and s.node(c.first_node_id).ordinal == c.introduced_ordinal
    assert {r.id for r in system.structure_store.nodes("book", "section")} == {x.id for x in s.sections()}


# ----------------------------------------------------------------------------- no global lock
def test_questions_are_answered_while_another_document_is_still_being_processed(make, book, tmp_path):
    brain = Brain()
    s = make(brain)
    s.ingest(book, "book")                                                                       # document A is ready
    other = tmp_path / "other.md"
    other.write_text("# Other\n\nA different document about geology and rocks.\n\n## More\n\nMore text about rocks.\n", encoding="utf-8")
    brain.gate = threading.Event()                                                               # B's knowledge build will block
    t = threading.Thread(target=lambda: s.ingest(other, "other"), daemon=True)
    t.start()
    deadline = time.time() + 5
    while not any("geology" in u for _, u in brain.calls) and time.time() < deadline:
        time.sleep(0.02)
    started = time.time()
    brain.gate.clear()
    answer = s.tutor.ask("u", "book", "What is the Calvin cycle?", ordinal(s, "s1.2"))            # served while B is mid-ingest
    # (the same brain is gated for extraction only; tutor requests are not)
    assert answer.text.startswith("LEVEL=") and time.time() - started < 3
    brain.gate.set()
    t.join(10)
    assert not t.is_alive()
