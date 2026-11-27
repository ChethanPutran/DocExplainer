"""Learn tab, feedback bar, suggested resources, and the window manager calls behind them (real SDK, mock server)."""
import json
from types import SimpleNamespace

import httpx
import pytest

from doc_explainer_ui.gui.managers.window_manager import WindowManager
from doc_explainer_ui.gui.utils.background import Background
from doc_explainer_ui.gui.widgets.sidebar.feedback_bar import FeedbackBar
from doc_explainer_ui.gui.widgets.sidebar.learn_panel import OVERVIEW, QUIZ, SETUP, LearnPanel
from doc_explainer_ui.gui.widgets.sidebar.recommendations_panel import RecommendationsPanel, parse_doc_url
from doc_explainer_ui.gui.widgets.sidebar.sidebar import Sidebar

from doc_explainer_sdk import DocExplainerClient

Q = [{"id": f"q{i}", "concept_id": "c", "concept": "ATP", "question": f"Q{i}?", "options": ["a", "b", "c"],
      "difficulty": 0.5, "kind": "practice"} for i in range(2)]


@pytest.fixture
def panel(qapp):
    return LearnPanel()


def test_learn_panel_shows_build_progress_then_status(panel):
    panel.set_document("d1")
    panel.set_building(0.42, "reading section 2 of 5")
    assert not panel.build_bar.isHidden() and panel.build_bar.value() == 42 and "section 2" in panel.stage_label.text()
    assert not panel.practice_button.isEnabled()
    panel.set_knowledge("ready", 5, 21)
    assert panel.build_bar.isHidden() and "21 concepts in 5 sections" in panel.status_label.text()
    assert panel.practice_button.isEnabled() and panel.diagnostic_button.isEnabled()


@pytest.mark.parametrize("status", ["failed", "none"])
def test_quizzes_are_disabled_without_a_knowledge_graph(panel, status):
    panel.set_knowledge(status, 3, 0)
    assert not panel.practice_button.isEnabled() and "Quizzes need" in panel.message.text()


def test_learner_summary_lists_strong_and_weak_concepts(panel):
    panel.set_learner("beginner", [
        {"name": "ATP", "p_known": 0.9, "evidence": 3}, {"name": "sugar", "p_known": 0.2, "evidence": 2},
        {"name": "unseen", "p_known": 0.1, "evidence": 0}])
    assert "beginner" in panel.level_label.text()
    assert "ATP" in panel.strong_label.text() and "sugar" in panel.weak_label.text() and "unseen" not in panel.weak_label.text()


def test_diagnostic_form_emits_level_and_topics(panel):
    got = []
    panel.diagnostic_requested.connect(lambda level, topics: got.append((level, topics)))
    panel.set_knowledge("ready", 2, 4)
    panel.diagnostic_button.click()
    assert panel.stack.currentIndex() == SETUP
    panel.level_combo.setCurrentText("beginner")
    panel.topics_edit.setText("photosynthesis, , cells ")
    panel.stack.currentWidget().findChildren(type(panel.diagnostic_button))[0].click()   # Start
    assert got == [("beginner", ["photosynthesis", "cells"])] and panel.stack.currentIndex() == OVERVIEW


def test_quiz_flow_answers_next_and_finish(panel):
    answers, finished = [], []
    panel.answer_submitted.connect(lambda q, c: answers.append((q, c)))
    panel.quiz_finished.connect(finished.append)
    panel.show_questions(Q, "diagnostic")
    assert panel.stack.currentIndex() == QUIZ and "question 1 of 2" in panel.quiz_title.text()
    panel.submit_button.click()                                           # nothing picked
    assert answers == [] and "Pick an answer" in panel.result_label.text()
    panel.group.button(1).setChecked(True)
    panel.submit_button.click()
    assert answers == [("q0", 1)] and not panel.submit_button.isEnabled()
    panel.show_answer({"correct": False, "correct_index": 2, "explanation": "because"})
    assert "Not quite" in panel.result_label.text() and "because" in panel.result_label.text()
    assert not panel.next_button.isHidden() and panel.next_button.text() == "Next"
    assert all(not b.isEnabled() for b in panel.group.buttons())
    panel.next_button.click()
    assert "question 2 of 2" in panel.quiz_title.text()
    panel.group.button(0).setChecked(True)
    panel.submit_button.click()
    panel.show_answer({"correct": True, "correct_index": 0, "explanation": ""})
    assert panel.next_button.text() == "Finish" and "Correct" in panel.result_label.text()
    panel.next_button.click()
    assert finished == ["diagnostic"] and panel.stack.currentIndex() == OVERVIEW


def test_leaving_before_answering_reports_nothing_and_no_questions_is_explained(panel):
    finished = []
    panel.quiz_finished.connect(finished.append)
    panel.show_questions(Q, "practice")
    panel.leave_button.click()
    assert finished == [] and panel.stack.currentIndex() == OVERVIEW
    panel.show_questions([], "practice")
    assert "No questions" in panel.message.text()


def test_answer_error_lets_the_reader_retry(panel):
    panel.show_questions(Q, "practice")
    panel.group.button(0).setChecked(True)
    panel.submit_button.click()
    panel.show_answer_error("The server is busy")
    assert panel.submit_button.isEnabled() and "busy" in panel.result_label.text()


def test_feedback_bar_gives_one_verdict(qapp):
    bar, got = FeedbackBar(), []
    bar.verdict_given.connect(got.append)
    assert not bar.buttons["too_hard"].isEnabled()
    bar.reset(True)
    bar.buttons["too_hard"].click()
    assert got == ["too_hard"] and not bar.buttons["too_easy"].isEnabled()


def test_sidebar_sends_feedback_with_the_concepts_of_the_answer_and_fills_resources(qapp):
    sb, got = Sidebar(), []
    sb.feedback_given.connect(lambda v, ids: got.append((v, ids)))
    result = {"text": "t", "explanation": {"context_used": {"concept_ids": ["c1", "c2"]},
                                           "resources": [{"title": "A", "url": "https://x.org", "type": "video", "source": "YouTube"}]}}
    sb.on_reply_done(result)
    sb.feedback_bar.buttons["too_easy"].click()
    assert got == [("too_easy", ["c1", "c2"])]
    assert any(w.text() == "Open resource" for w in sb.recommendations_panel.findChildren(type(sb.feedback_bar.buttons["too_easy"])))
    sb.on_reply_done({"text": "t", "explanation": {"context_used": {}}})
    assert not sb.feedback_bar.buttons["too_easy"].isEnabled()           # nothing to attach feedback to


def test_resources_distinguish_document_sections_from_links(qapp):
    assert parse_doc_url("doc://d1#sec-3") == "sec-3" and parse_doc_url("https://x.org/a#b") is None
    panel, sections, concepts = RecommendationsPanel(), [], []
    panel.section_requested.connect(sections.append)
    panel.concept_requested.connect(concepts.append)
    panel.set_resources([{"title": "Photosynthesis > Calvin cycle", "url": "doc://d1#sec-3", "type": "section",
                          "description": "Introduces sugar (page 4)", "difficulty": "beginner"},
                         {"title": "Web", "url": "https://x.org", "type": "article"}])
    panel.set_review([{"name": "ATP", "p_known": 0.25, "evidence": 2}, {"name": "sugar", "p_known": 0.9, "evidence": 2},
                      {"name": "new", "p_known": 0.1, "evidence": 0}])
    buttons = {b.text(): b for b in panel.findChildren(type(panel.refresh_button))}
    assert {"Go to section", "Open resource", "Explain this"} <= set(buttons)
    buttons["Go to section"].click()
    buttons["Explain this"].click()
    assert sections == ["sec-3"] and concepts == ["ATP"]          # strong and unseen concepts are not suggested
    panel.clear()
    assert panel._resources == [] and panel._review == []


# ----------------------------------------------------------------------------- window manager
DOC = {"doc_id": "d1", "title": "T", "filename": "f.md", "size_bytes": 3, "uploaded_at": "now",
       "knowledge": "ready", "sections": 4, "concepts": 8}
LEARNER = {"level": "beginner", "prior": 0.2, "position": None,
           "concepts": [{"concept_id": "c1", "name": "ATP", "p_known": 0.2, "evidence": 2}]}
STRUCT = {"doc_id": "d1", "title": "T", "format": "md", "pages": 5, "sections": [
    {"id": "sec-3", "ordinal": 4, "title": "Calvin cycle", "depth": 2, "page_start": 4, "page_end": 5}]}


class Win:
    def __init__(self):
        self.sidebar = Sidebar()
        self.messages, self.jumps = [], []

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda m, ms=0: self.messages.append(m))

    def show_section(self, doc_id, page, title=""):
        self.jumps.append((doc_id, page, title))


def make(handler):
    wm = object.__new__(WindowManager)
    wm.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    wm.client = DocExplainerClient("http://srv", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    wm._bg, wm._registered, wm._structures, wm.main_window = Background(), {"d1"}, {}, Win()
    return wm


class Srv:
    def __init__(self, **over):
        self.requests, self.over = [], over

    def __call__(self, req):
        path, body = req.url.path, (json.loads(req.content) if req.content else {})
        self.requests.append((req.method, path, body))
        if path in self.over:
            return self.over[path](req)
        table = {
            "/v1/documents/d1": DOC, "/v1/documents/d1/learner": LEARNER, "/v1/documents/d1/structure": STRUCT,
            "/v1/documents/d1/diagnostic": {"level": "beginner", "prior": 0.2, "questions": Q},
            "/v1/documents/d1/diagnostic/finish": {"level": "beginner", "prior": 0.3, "questions": []},
            "/v1/documents/d1/quiz": {"questions": Q},
            "/v1/documents/d1/quiz/q0/answer": {"correct": True, "correct_index": 1, "explanation": "yes"},
        }
        return httpx.Response(200, json=table.get(path, {"ok": True}))


def seen(srv, suffix):
    return [r for r in srv.requests if r[1].endswith(suffix)]


def test_showing_a_document_fills_status_level_and_review(pump):
    srv = Srv()
    wm = make(srv)
    wm.show_document("d1")
    panel = wm.main_window.sidebar.learn_panel
    assert pump(lambda: "beginner" in panel.level_label.text())
    assert "8 concepts in 4 sections" in panel.status_label.text()
    recs = wm.main_window.sidebar.recommendations_panel
    assert any(w.text().startswith("ATP") for w in recs.findChildren(type(panel.status_label)))


def test_a_document_that_is_not_ready_is_not_queried(pump):
    srv = Srv()
    wm = make(srv)
    wm._registered.clear()
    wm.show_document("d1")
    assert srv.requests == []


def test_results_for_a_document_the_user_left_are_dropped(pump):
    wm = make(Srv())
    wm.show_document("d1")
    wm.main_window.sidebar.learn_panel.set_document("other")     # user switched tabs
    wm.refresh_learning("d1")
    pump(lambda: wm._bg.active == 0)
    assert wm.main_window.sidebar.learn_panel.level_label.text() == ""


def test_diagnostic_then_answer_then_finish_reaches_the_service(pump):
    srv = Srv()
    wm = make(srv)
    panel = wm.main_window.sidebar.learn_panel
    wm.show_document("d1")
    assert pump(lambda: "beginner" in panel.level_label.text())
    wm.start_diagnostic("d1", "beginner", ["cells"])
    assert pump(lambda: panel.stack.currentIndex() == QUIZ)
    assert seen(srv, "/diagnostic")[0][2] == {"level": "beginner", "known_topics": ["cells"], "questions": 6}
    wm.answer_quiz("d1", "q0", 1)
    assert pump(lambda: "Correct" in panel.result_label.text())
    assert seen(srv, "/quiz/q0/answer")[0][2] == {"choice": 1}
    wm.quiz_finished("d1", "diagnostic")
    assert pump(lambda: seen(srv, "/diagnostic/finish"))
    assert any("level was updated" in m for m in wm.main_window.messages) or pump(lambda: any("level was updated" in m for m in wm.main_window.messages))


def test_practice_quiz_uses_the_requested_section(pump):
    srv = Srv()
    wm = make(srv)
    wm.show_document("d1")
    pump(lambda: srv.requests)
    wm.practice_quiz("d1", 4)
    assert pump(lambda: wm.main_window.sidebar.learn_panel.stack.currentIndex() == QUIZ)
    assert seen(srv, "/quiz")[0][2] == {"section_id": 4, "count": 5}


def test_a_failed_quiz_request_is_shown_not_raised(pump):
    busy = {"/v1/documents/d1/quiz": lambda r: httpx.Response(503, json={"error": {"code": "queue_full", "message": "Try later"}})}
    wm = make(Srv(**busy))
    wm.show_document("d1")
    wm.practice_quiz("d1")
    panel = wm.main_window.sidebar.learn_panel
    assert pump(lambda: "Try later" in panel.message.text())


def test_feedback_posts_verdict_and_ids_and_ignores_empty(pump):
    srv = Srv()
    wm = make(srv)
    wm.send_feedback("d1", "too_hard", [])
    wm.send_feedback("d1", "too_hard", ["c1"])
    assert pump(lambda: seen(srv, "/feedback"))
    assert [r[2] for r in seen(srv, "/feedback")] == [{"verdict": "too_hard", "concept_ids": ["c1"]}]


def test_going_to_a_resource_section_scrolls_to_its_page(pump):
    srv = Srv()
    wm = make(srv)
    wm.go_to_section("d1", "sec-3")
    assert pump(lambda: wm.main_window.jumps == [("d1", 4, "Calvin cycle")])
    wm.go_to_section("d1", "sec-3")                               # structure is cached
    assert pump(lambda: len(wm.main_window.jumps) == 2) and len(seen(srv, "/structure")) == 1
    wm.go_to_section("d1", "gone")
    assert pump(lambda: any("no longer" in m for m in wm.main_window.messages))


def test_mark_progress_sends_the_section(pump):
    srv = Srv()
    wm = make(srv)
    wm.mark_progress("d1", 4)
    assert pump(lambda: seen(srv, "/progress")) and seen(srv, "/progress")[0][2] == {"section_id": 4}


def test_the_sidebar_can_be_cleared_repeatedly(qapp):
    """Switching tabs clears the sidebar every time; widgets deleted by one clear must not be reused by the next."""
    sb = Sidebar()
    for _ in range(3):
        sb.clear()
        qapp.processEvents()
    sb.chat_panel.add_user_message("hello")
    sb.chat_panel.begin_reply()
    sb.clear()
    qapp.processEvents()
    sb.clear()
    qapp.processEvents()
    assert sb.chat_panel.messages() == []


# ----------------------------------------------------------------------------- job status
from doc_explainer_ui.gui.windows.jobs_window import JobStatusWindow, describe  # noqa: E402

JOBS = [
    {"job_id": "j1", "state": "running", "filename": "paper.pdf", "created_at": "2026-11-27T10:00:00", "started_at": "2026-11-27T10:00:05",
     "progress": 0.42, "stage": "reading section 3 of 8", "kind": "register_document"},
    {"job_id": "j2", "state": "queued", "filename": "book.pdf", "created_at": "2026-11-27T10:01:00", "queue_position": 2, "progress": 0.0},
    {"job_id": "j3", "state": "failed", "filename": "bad.pdf", "created_at": "2026-11-27T09:00:00", "finished_at": "2026-11-27T09:00:09",
     "error": {"code": "processing_failed", "message": "cannot parse bad.pdf"}},
    {"job_id": "j4", "state": "succeeded", "filename": "ok.md", "created_at": "2026-11-27T08:00:00", "progress": 1.0,
     "document": {"doc_id": "d9", "title": "Rocks", "filename": "ok.md", "size_bytes": 1234, "uploaded_at": "x",
                  "knowledge": "partial", "sections": 4, "concepts": 7}},
]


def test_job_details_are_complete_and_readable():
    running, queued, failed, done = (describe(j) for j in JOBS)
    assert "Processing" in running and "42%" in running and "reading section 3 of 8" in running and "Started:" in running
    assert "Waiting" in queued and "Queue position:  2" in queued
    assert "What went wrong" in failed and "cannot parse bad.pdf" in failed and "processing_failed" in failed
    assert "Concepts found:  7" in done and "partly built" in done and "d9" in done and "1,234 bytes" in done
    assert "explanations are adapted to you only where the graph exists" in done


def test_job_window_lists_newest_first_and_shows_the_selected_jobs_details(qapp):
    w = JobStatusWindow()
    assert "No documents" in w.status.text()
    w.set_jobs(JOBS)
    assert [w.table.item(r, 0).text() for r in range(4)] == ["book.pdf", "paper.pdf", "bad.pdf", "ok.md"]
    assert w.table.item(0, 1).text() == "Waiting (#2)" and w.table.item(1, 2).text() == "42%" and w.table.item(3, 2).text() == "100%"
    assert w.table.item(2, 3).text() == "cannot parse bad.pdf"                  # the reason, not an empty cell
    assert "2 active, 4 total" in w.status.text()
    w.table.selectRow(2)
    assert "What went wrong" in w.details.toPlainText()
    updated = [dict(j, progress=0.6, stage="indexing") if j["job_id"] == "j1" else j for j in JOBS]
    w.set_jobs(updated)                                                           # a refresh keeps the selection
    assert w._selected == "j3" and "What went wrong" in w.details.toPlainText()
    w.set_error("Cannot reach the server")
    assert "Cannot reach" in w.status.text()


def test_job_window_refreshes_itself_only_while_visible(qapp, pump):
    w, n = JobStatusWindow(), []
    w.REFRESH_MS = 20
    w.refresh_requested.connect(lambda: n.append(1))
    w.show()
    w._timer.start(20)
    assert pump(lambda: len(n) >= 2)
    w.hide()
    count = len(n)
    pump(lambda: False, timeout=0.2)
    assert len(n) == count and not w._timer.isActive()


def test_show_jobs_fetches_and_fills_the_window_without_stacking_requests(pump):
    def jobs(req):
        return httpx.Response(200, json={"jobs": JOBS})
    srv = Srv(**{"/v1/jobs": jobs})
    wm = make(srv)
    wm.show_jobs()
    wm.refresh_jobs()                                                              # while one is in flight: ignored
    assert pump(lambda: wm.jobs_window.table.rowCount() == 4)
    assert len(seen(srv, "/jobs")) == 1
    assert wm.jobs_window.table.item(1, 0).text() == "paper.pdf"
    wm.refresh_jobs()
    assert pump(lambda: len(seen(srv, "/jobs")) == 2)


def test_a_server_error_is_shown_in_the_job_window(pump):
    wm = make(Srv(**{"/v1/jobs": lambda r: httpx.Response(500, json={"error": {"code": "internal_error", "message": "boom"}})}))
    wm.show_jobs()
    assert pump(lambda: "boom" in wm.jobs_window.status.text())
    assert wm._jobs_refreshing is False


def test_the_sidebar_offers_job_details_while_the_ai_is_unavailable(qapp):
    sb, hits = Sidebar(), []
    sb.details_requested.connect(lambda: hits.append(1))
    sb.set_unavailable("Document is being processed: 10%")
    assert not sb.details_button.isHidden()
    sb.details_button.click()
    assert hits == [1]
    sb.set_available()
    assert sb.details_button.isHidden()


def test_a_late_status_refresh_does_not_wipe_a_quiz_error(panel):
    panel.set_knowledge("ready", 4, 8)
    panel.set_message("The server is busy, try later")
    panel.set_knowledge("ready", 4, 8)                          # a refresh that finished after the error
    assert "busy" in panel.message.text()
    panel.set_knowledge("failed", 4, 0)
    assert "Quizzes need" in panel.message.text()
    panel.set_knowledge("ready", 4, 8)
    assert panel.message.text() == ""                           # its own notice goes away once it no longer applies


def test_the_quiz_tab_is_named_and_the_toolbar_opens_it(qapp):
    from types import SimpleNamespace as NS
    from doc_explainer_ui.config import UIConfig
    from doc_explainer_ui.gui.models.signals import UISignals
    from doc_explainer_ui.gui.windows.main_window import MainWindow
    cap = {}
    wm = NS(register_document_async=lambda path, on_done=None, on_error=None, on_progress=None, on_cancelled=None, build_graph=True: cap.update(done=on_done),
            show_document=lambda d: None)
    cfg = UIConfig()
    cfg.voice.enabled = False
    w = MainWindow(wm, NS(toggle_theme=lambda: None), NS(register_action=lambda *a, **k: None), None, UISignals(), cfg)
    names = [w.sidebar.tab_widget.tabText(i) for i in range(w.sidebar.tab_widget.count())]
    assert "Learn && Quiz" in names                               # "&&" is Qt's escape: it displays "Learn & Quiz"
    w.toolbar.quiz_action.trigger()                                  # nothing open: explained, not silent
    assert "once the document has been processed" in w.statusBar().currentMessage()
    import os, tempfile
    p = os.path.join(tempfile.mkdtemp(), "n.txt")
    open(p, "w").write("text")
    w.open_document(p)
    w.toolbar.quiz_action.trigger()                                  # still processing
    assert "once the document" in w.statusBar().currentMessage()
    cap["done"]("d1")
    w.toolbar.quiz_action.trigger()
    assert w.sidebar.tab_widget.currentWidget() is w.sidebar.learn_panel
