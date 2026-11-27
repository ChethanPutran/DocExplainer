"""The chat panel: streaming, status, follow-ups at the end of the answer, typing and speaking, sessions, notes."""
import threading
import time
from types import SimpleNamespace

import pymupdf
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal

from doc_explainer_ui.gui.widgets import PDFViewer, Sidebar, TextViewer
from doc_explainer_ui.gui.widgets.sidebar.chat_panel import AUTOSEND_PAUSE_MS, ChatPanel, split_sentences
from doc_explainer_ui.gui.widgets.sidebar.recommendations_panel import RecommendationsPanel
from doc_explainer_ui.gui.widgets.voice.sentence_speaker import SentenceSpeaker
from doc_explainer_ui.gui.windows.jobs_window import JobStatusWindow


class FakeSpeaker(QObject):
    """Records what it is asked to say; the test decides when each sentence 'starts' and when speech ends."""
    sentence_started = Signal(str)
    idle = Signal()
    unavailable = Signal(str)

    def __init__(self):
        super().__init__()
        self.said, self.stopped, self._busy = [], 0, False

    @property
    def speaking(self):
        return self._busy

    def enqueue(self, s):
        self.said.append(s)
        self._busy = True

    def stop(self):
        self.stopped += 1
        self._busy = False
        self.said = []

    def start(self, i):
        self.sentence_started.emit(self.said[i])

    def finish(self):
        self._busy = False
        self.idle.emit()


RESULT = {"text": "Final text.", "explanation": {"follow_up_questions": ["Why?", "How?"], "known_concepts_used": ["ATP"],
                                                 "unknown_concepts_explained": ["sugar"]}}


@pytest.fixture
def panel(qapp):
    return ChatPanel()


@pytest.fixture
def spoken(qapp):
    sp = FakeSpeaker()
    p = ChatPanel(speaker=sp)
    p.speaker_button.setChecked(True)
    return p, sp


# ----------------------------------------------------------------------------- streaming and status
def test_a_reply_grows_as_deltas_arrive_with_a_status_line_then_completes(panel):
    panel.add_user_message("What is ATP?")
    panel.begin_reply()
    panel.set_status("Understanding your request")
    assert panel._busy and panel.send_button.text() == "Stop" and panel.status_label.text() == "Understanding your request"
    panel.append_delta("ATP is ")
    panel.append_delta("the energy currency.")
    assert panel.messages()[1].text == "ATP is the energy currency."                            # visible while still streaming
    panel.finish_reply({"text": "ATP is the energy currency.", "explanation": RESULT["explanation"]})
    assert not panel._busy and panel.send_button.text() == "Send" and panel.status_label.text() == ""
    assert [m.role for m in panel.messages()] == ["user", "assistant"]


def test_follow_up_questions_come_at_the_end_of_the_answer_and_clicking_one_asks_it(panel):
    asked = []
    panel.send_requested.connect(asked.append)
    panel.begin_reply()
    panel.append_delta("Answer.")
    assert panel.messages()[0].follow_up_buttons() == []                                         # not before the answer is complete
    panel.finish_reply(RESULT)
    bubble = panel.messages()[0]
    assert bubble.text == "Final text."
    questions = [b for b in bubble.follow_up_buttons() if b.text() in ("Why?", "How?")]
    assert [b.text() for b in questions] == ["Why?", "How?"]
    questions[1].click()
    assert asked == ["How?"]
    assert not any(w.text() == "Follow-up" for w in panel.parent().findChildren(QObject)) if panel.parent() else True


def test_there_is_no_separate_follow_up_tab(qapp):
    sb = Sidebar()
    names = [sb.tab_widget.tabText(i) for i in range(sb.tab_widget.count())]
    assert names == ["Explanation", "Resources", "Learn && Quiz"]


def test_a_failure_replaces_the_pending_answer_and_frees_the_input(panel):
    panel.begin_reply()
    panel.append_delta("Half an ans")
    panel.fail_reply("The model is down")
    assert "Half an ans" in panel.messages()[0].text and "The model is down" in panel.messages()[0].text
    assert not panel._busy and panel.status_label.text() == ""


def test_stop_keeps_the_partial_answer(panel):
    panel.begin_reply()
    panel.append_delta("So far")
    panel.stop_reply()
    assert "So far" in panel.messages()[0].text and "stopped" in panel.messages()[0].text and not panel._busy


# ----------------------------------------------------------------------------- typing
def test_enter_sends_shift_enter_does_not_and_the_box_clears(panel, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    sent = []
    panel.send_requested.connect(sent.append)
    panel.input.setPlainText("Why is the sky blue?")
    panel.input.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier))
    assert sent == []
    panel.input.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert sent == ["Why is the sky blue?"] and panel.input.toPlainText() == ""
    panel.send_button.click()                                                                    # empty: nothing sent
    assert sent == ["Why is the sky blue?"]


def test_while_a_reply_is_being_written_the_send_button_stops_it(panel):
    stops = []
    panel.stop_requested.connect(lambda: stops.append(1))
    panel.begin_reply()
    panel.send_button.click()
    assert stops == [1]


# ----------------------------------------------------------------------------- the microphone
def test_spoken_words_land_in_the_input_box_for_the_reader_to_edit_and_send(panel):
    sent = []
    panel.send_requested.connect(sent.append)
    panel.add_transcription("what is the")
    panel.add_transcription("calvin cycle")
    assert panel.input.toPlainText() == "what is the calvin cycle" and sent == []                 # nothing sent behind their back


def test_auto_send_sends_after_a_pause_but_not_while_they_keep_talking(panel, pump):
    sent = []
    panel.send_requested.connect(sent.append)
    panel.autosend.setChecked(True)
    panel.add_transcription("what is the")
    panel._autosend_timer.setInterval(60)
    panel._autosend_timer.start(60)
    time.sleep(0.03)
    panel.add_transcription("calvin cycle")                                                      # more speech: the countdown restarts
    assert sent == []
    assert pump(lambda: sent == ["what is the calvin cycle"], timeout=3)
    assert panel.input.toPlainText() == ""
    assert AUTOSEND_PAUSE_MS >= 1000                                                             # a real pause, not a breath


def test_without_auto_send_nothing_is_ever_sent_by_voice(panel, pump):
    sent = []
    panel.send_requested.connect(sent.append)
    panel.add_transcription("hello there")
    pump(lambda: False, timeout=0.2)
    assert sent == [] and not panel._autosend_timer.isActive()


def test_the_tutors_own_voice_is_not_taken_for_a_question(spoken):
    p, sp = spoken
    sp._busy = True
    p.add_transcription("the tutor speaking")
    assert p.input.toPlainText() == ""


def test_the_mic_button_reports_toggles_and_shows_the_real_state(panel):
    toggles = []
    panel.mic_toggled.connect(toggles.append)
    panel.mic_button.click()
    panel.mic_button.click()
    assert toggles == [True, False]
    panel.set_listening(True)
    assert panel.mic_button.isChecked() and panel.mic_button.text() == "🔴" and toggles == [True, False]   # no echo signal
    panel.set_listening(False)
    assert not panel.mic_button.isChecked()


# ----------------------------------------------------------------------------- the speaker: text appears as it is spoken
def test_sentences_are_revealed_only_when_each_starts_being_spoken(spoken):
    p, sp = spoken
    p.begin_reply()
    p.append_delta("First sentence. Second sentence. Thi")
    assert sp.said == ["First sentence. ", "Second sentence. "]                                   # complete sentences go to the voice
    assert p.messages()[0].text == ""                                                              # nothing shown before it is spoken
    sp.start(0)
    assert p.messages()[0].text == "First sentence. "
    p.append_delta("rd one.")
    sp.start(1)
    assert p.messages()[0].text == "First sentence. Second sentence. "
    p.finish_reply({"text": "First sentence. Second sentence. Third one.", "explanation": {"follow_up_questions": ["Why?"]}})
    assert sp.said[-1] == "Third one." and p.status_label.text() == "Speaking..."
    assert [b for b in p.messages()[0].follow_up_buttons() if b.text() == "Why?"] == []           # follow-ups wait for the voice
    sp.start(2)
    assert p.messages()[0].text.endswith("Third one.")
    sp.finish()
    assert p.messages()[0].text == "First sentence. Second sentence. Third one."
    assert [b.text() for b in p.messages()[0].follow_up_buttons() if b.text() == "Why?"] == ["Why?"]
    assert not p._busy


def test_with_the_speaker_off_text_is_shown_at_once_and_nothing_is_spoken(spoken):
    p, sp = spoken
    p.speaker_button.setChecked(False)
    p.begin_reply()
    p.append_delta("Hello world. More text")
    assert p.messages()[0].text == "Hello world. More text" and sp.said == []


def test_turning_the_speaker_off_mid_reply_stops_the_voice_and_shows_the_rest(spoken):
    p, sp = spoken
    p.begin_reply()
    p.append_delta("One. Two. Three.")
    sp.start(0)
    p.speaker_button.setChecked(False)
    assert sp.stopped >= 1
    assert p.messages()[0].text == "One. Two. Three."
    p.append_delta(" Four.")
    assert p.messages()[0].text.endswith("Four.")                                                  # and carries on silently


def test_no_speech_engine_falls_back_to_showing_text_and_disables_the_button(spoken):
    p, sp = spoken
    p.begin_reply()
    p.append_delta("Hello there. How are you")
    sp.unavailable.emit("no engine")
    assert p.messages()[0].text == "Hello there. How are you"
    assert not p.speaker_button.isEnabled() and "not available" in p.speaker_button.toolTip()


def test_a_question_asked_while_speaking_is_not_blocked_by_a_finished_reply(spoken):
    p, sp = spoken
    p.begin_reply()
    p.append_delta("Short.")
    p.finish_reply({"text": "Short.", "explanation": {}})
    sp.start(0)
    sp.finish()
    p.begin_reply()                                                                                # the next turn starts cleanly
    assert p._awaiting == [] and p._reply is not None


def test_sentence_splitting():
    assert split_sentences("A b. C d! E? f") == (["A b. ", "C d! ", "E? "], "f")
    assert split_sentences("no end") == ([], "no end")
    assert split_sentences("line one\nline two") == (["line one\n"], "line two")
    assert split_sentences("3.14 is pi. ok")[0] == ["3.14 is pi. "]                                 # a decimal point is not a sentence end


# ----------------------------------------------------------------------------- sessions and history
SESSIONS = [{"id": "s2", "doc_id": "d", "title": "Exam prep", "messages": 4}, {"id": "s1", "doc_id": "d", "title": "First", "messages": 2}]


def test_sessions_are_listed_selected_and_managed(panel, monkeypatch):
    got = {}
    panel.session_selected.connect(lambda s: got.update(select=s))
    panel.new_session_requested.connect(lambda: got.update(new=True))
    panel.rename_session_requested.connect(lambda s, t: got.update(rename=(s, t)))
    panel.delete_session_requested.connect(lambda s: got.update(delete=s))
    panel.set_sessions(SESSIONS, "s2")
    assert [panel.session_combo.itemText(i) for i in range(2)] == ["Exam prep  (2)", "First  (1)"]
    assert panel.session_combo.currentData() == "s2"
    panel.session_combo.activated.emit(1)
    assert got["select"] == "s1"
    panel.session_combo.activated.emit(0)                                                          # the current one: no reload
    panel.new_button.click()
    assert got["new"]
    from doc_explainer_ui.gui.widgets.sidebar import chat_panel as cp
    monkeypatch.setattr(cp.QInputDialog, "getText", staticmethod(lambda *a, **k: ("Chapter 2", True)))
    panel.rename_button.click()
    assert got["rename"] == ("s2", "Chapter 2")
    monkeypatch.setattr(cp.QMessageBox, "question", staticmethod(lambda *a, **k: cp.QMessageBox.StandardButton.Yes))
    panel.delete_button.click()
    assert got["delete"] == "s2"
    monkeypatch.setattr(cp.QMessageBox, "question", staticmethod(lambda *a, **k: cp.QMessageBox.StandardButton.No))
    got.pop("delete")
    panel.delete_button.click()
    assert "delete" not in got


def test_a_past_session_is_shown_with_both_sides_and_its_follow_ups(panel):
    panel.show_history([{"role": "user", "text": "What is ATP?", "meta": {}},
                        {"role": "assistant", "text": "The energy currency.", "meta": {"follow_ups": ["Where is it made?"]}}],
                       [{"id": "n1", "kind": "note", "text": "remember"}])
    assert [(m.role, m.text) for m in panel.messages()] == [("user", "What is ATP?"), ("assistant", "The energy currency.")]
    assert [b.text() for b in panel.messages()[1].follow_up_buttons() if b.text() == "Where is it made?"] == ["Where is it made?"]
    assert "(1)" in panel.notes_button.text()
    panel.show_history([], [])
    assert panel.messages() == [] and panel.empty_label is not None


def test_switching_session_during_a_reply_does_not_leak_the_old_stream_into_the_new_one(panel):
    panel.begin_reply()
    panel.append_delta("old reply")
    panel.show_history([{"role": "user", "text": "new session question", "meta": {}}], [])
    panel.append_delta(" still arriving")
    assert [m.text for m in panel.messages()] == ["new session question"]


# ----------------------------------------------------------------------------- notes
def test_notes_can_be_added_edited_deleted_and_the_session_summarised(panel, monkeypatch):
    got = {}
    panel.note_add_requested.connect(lambda t: got.update(add=t))
    panel.note_edit_requested.connect(lambda i, t: got.update(edit=(i, t)))
    panel.note_delete_requested.connect(lambda i: got.update(delete=i))
    panel.summarize_requested.connect(lambda: got.update(summarize=True))
    panel.notes_button.setChecked(True)
    assert panel.pages.currentIndex() == 1
    panel.note_input.setPlainText("ATP is the energy currency")
    panel.add_note_button.click()
    assert got["add"] == "ATP is the energy currency" and panel.note_input.toPlainText() == ""
    panel.add_note_button.click()                                                                  # empty: ignored
    panel.set_notes([{"id": "n1", "kind": "note", "text": "mine"}, {"id": "n2", "kind": "summary", "text": "- gist"}])
    assert len(panel.note_cards()) == 2
    from PySide6.QtWidgets import QToolButton
    from doc_explainer_ui.gui.widgets.sidebar import chat_panel as cp
    monkeypatch.setattr(cp.QInputDialog, "getMultiLineText", staticmethod(lambda *a, **k: ("edited", True)))
    buttons = {b.text(): b for b in panel.note_cards()[0].findChildren(QToolButton)}
    buttons["Edit"].click()
    buttons["Delete"].click()
    assert got["edit"] == ("n1", "edited") and got["delete"] == "n1"
    panel.summarize_button.click()
    assert got["summarize"]
    panel.notes_button.setChecked(False)
    assert panel.pages.currentIndex() == 0


def test_a_reply_can_be_saved_to_the_notes(panel):
    saved = []
    panel.note_add_requested.connect(saved.append)
    panel.begin_reply()
    panel.append_delta("Worth keeping.")
    panel.finish_reply({"text": "Worth keeping.", "explanation": {}})
    from PySide6.QtWidgets import QToolButton
    next(b for b in panel.messages()[0].findChildren(QToolButton) if "Save to notes" in b.text()).click()
    assert saved == ["Worth keeping."]


# ----------------------------------------------------------------------------- resources
def test_explain_this_disappears_after_it_is_clicked_and_stays_gone_for_that_document(qapp):
    panel, asked = RecommendationsPanel(), []
    panel.concept_requested.connect(asked.append)
    panel.set_document("d1")
    beliefs = [{"name": "ATP", "p_known": 0.2, "evidence": 2}, {"name": "sugar", "p_known": 0.3, "evidence": 2}]
    panel.set_review(beliefs)

    def explain_buttons():
        for _ in range(3):
            qapp.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)       # the event loop does this in the real app
        return [b for b in panel.findChildren(type(panel.refresh_button)) if b.text() == "Explain this"]
    assert len(explain_buttons()) == 2
    explain_buttons()[0].click()
    assert asked == ["ATP"] and len(explain_buttons()) == 1                                         # that card is gone
    panel.set_review(beliefs)                                                                       # a refresh brings no ghost back
    assert len(explain_buttons()) == 1
    panel.set_document("d2")
    panel.set_review(beliefs)
    assert len(explain_buttons()) == 2                                                              # another document: fresh


def test_internet_resources_are_listed_separately_with_their_source(qapp):
    panel = RecommendationsPanel()
    panel.set_resources([
        {"title": "A > B", "url": "doc://d#s2.1", "type": "section", "source": "This document", "description": "x"},
        {"title": "Video lectures: ATP", "url": "https://www.youtube.com/results?search_query=ATP", "type": "video", "source": "YouTube",
         "description": "Lectures", "difficulty": "beginner"},
        {"title": "Encyclopedia: ATP", "url": "https://en.wikipedia.org/w/index.php?search=ATP", "type": "documentation", "source": "Wikipedia"}])
    from PySide6.QtWidgets import QLabel
    texts = [w.text() for w in panel.findChildren(QLabel)]
    assert "Read next in this document" in texts and "On the internet" in texts
    assert any("YouTube" in t and "video" in t for t in texts) and any("Wikipedia" in t for t in texts)
    buttons = [b.text() for b in panel.findChildren(type(panel.refresh_button))]
    assert buttons.count("Open resource") == 2 and buttons.count("Go to section") == 1


def test_go_to_section_emits_the_id_of_each_resource_not_just_the_first(qapp):
    panel, got = RecommendationsPanel(), []
    panel.section_requested.connect(got.append)
    panel.set_resources([{"title": f"S{i}", "url": f"doc://d#s{i}", "type": "section"} for i in (1, 2, 3)])
    for b in [b for b in panel.findChildren(type(panel.refresh_button)) if b.text() == "Go to section"]:
        b.click()
    assert got == ["s1", "s2", "s3"]


# ----------------------------------------------------------------------------- jumping to a section's heading
def test_pdf_jumps_to_the_heading_even_when_all_sections_share_a_page(qapp, tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, y in enumerate((80, 300, 520, 740)):
        page.insert_text((72, y), f"Section heading number {i + 1}", fontsize=16)
    doc.save(str(tmp_path / "a.pdf"))
    v = PDFViewer()
    v.resize(600, 300)
    v.show()
    assert v.load(str(tmp_path / "a.pdf"))
    for _ in range(5):
        qapp.processEvents()                                                                       # let the layout settle
    bar = v.scroll_area.verticalScrollBar()
    positions = []
    for title in ("Section heading number 1", "Section heading number 3", "Section heading number 4"):
        assert v.go_to_section(1, title)
        positions.append(bar.value())
    assert positions[0] < positions[1] < positions[2]                                               # each section lands somewhere different
    assert v.go_to_section(1, "Spacing  is   lost: SectionheadingNumber2".replace("Spacing  is   lost: ", ""))   # tolerant of lost spaces
    assert not v.go_to_section(9, "nothing")


def test_text_viewer_jumps_to_the_heading_text(qapp, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Intro\n" + "filler line\n" * 200 + "The Calvin cycle\nmore\n")
    v = TextViewer()
    v.resize(500, 200)
    v.show()
    v.load(str(f))
    qapp.processEvents()
    assert v.go_to_section(1, "The Calvin cycle") and v.text_edit.textCursor().selectedText() == "The Calvin cycle"
    assert v.text_edit.verticalScrollBar().value() > 0
    assert not v.go_to_section(1, "Not in the text")


# ----------------------------------------------------------------------------- the speaker itself
class FakeEngine:
    def __init__(self):
        self.spoken, self.props, self.stopped = [], {}, 0
        self.gate = None

    def setProperty(self, k, v):
        self.props[k] = v

    def say(self, s):
        self.pending = s

    def runAndWait(self):
        if self.gate:
            self.gate.wait(3)
        self.spoken.append(self.pending)

    def stop(self):
        self.stopped += 1
        if self.gate:
            self.gate.set()


def test_the_speaker_says_sentences_in_order_announcing_each_start_then_goes_idle(qapp, pump):
    engine, started, idles = FakeEngine(), [], []
    sp = SentenceSpeaker(lambda: engine, rate=170, volume=0.5)
    sp.sentence_started.connect(started.append)
    sp.idle.connect(lambda: idles.append(1))
    sp.enqueue("One.")
    sp.enqueue("Two.")
    sp.enqueue("   ")                                                                              # nothing to say
    assert pump(lambda: idles)
    assert started == ["One.", "Two."] and engine.spoken == ["One.", "Two."] and engine.props == {"rate": 170, "volume": 0.5}
    assert not sp.speaking
    sp.enqueue("Again.")                                                                           # reusable after going idle
    assert pump(lambda: engine.spoken[-1] == "Again.")


def test_stop_drops_the_queue_interrupts_the_current_sentence_and_says_idle(qapp, pump):
    engine, started, idles = FakeEngine(), [], []
    engine.gate = threading.Event()
    sp = SentenceSpeaker(lambda: engine)
    sp.sentence_started.connect(started.append)
    sp.idle.connect(lambda: idles.append(1))
    for s in ("A.", "B.", "C."):
        sp.enqueue(s)
    assert pump(lambda: started == ["A."])
    sp.stop()
    assert engine.stopped == 1 and not sp.speaking and idles
    time.sleep(0.2)
    assert started == ["A."]                                                                       # B and C were never spoken


def test_a_missing_speech_engine_is_reported_once_and_never_raises(qapp, pump):
    def broken():
        raise RuntimeError("espeak not installed")
    sp, why = SentenceSpeaker(broken), []
    sp.unavailable.connect(why.append)
    sp.enqueue("Hello.")
    assert pump(lambda: why) and "espeak" in why[0]
    sp.enqueue("More.")                                                                            # silently ignored afterwards
    assert not sp.speaking


# ----------------------------------------------------------------------------- job control buttons
JOBS = [{"job_id": "j1", "state": "running", "filename": "a.pdf", "created_at": "2026-01-02", "progress": 0.3},
        {"job_id": "j2", "state": "paused", "filename": "b.pdf", "created_at": "2026-01-01", "progress": 0.5},
        {"job_id": "j3", "state": "succeeded", "filename": "c.pdf", "created_at": "2025-12-31", "progress": 1.0},
        {"job_id": "j4", "state": "cancelled", "filename": "d.pdf", "created_at": "2025-12-30"}]


def test_job_buttons_match_the_selected_jobs_state_and_emit_its_id(qapp, monkeypatch):
    w, got = JobStatusWindow(), []
    w.pause_requested.connect(lambda j: got.append(("pause", j)))
    w.resume_requested.connect(lambda j: got.append(("resume", j)))
    w.cancel_requested.connect(lambda j: got.append(("cancel", j)))
    w.set_jobs(JOBS)
    states = {}
    for row in range(4):
        w.table.selectRow(row)
        states[w._selected] = (w.pause_button.isEnabled(), w.resume_button.isEnabled(), w.cancel_button.isEnabled())
    assert states == {"j1": (True, False, True), "j2": (False, True, True), "j3": (False, False, False), "j4": (False, False, False)}
    from doc_explainer_ui.gui.windows import jobs_window as jw
    w.table.selectRow(0)
    w.pause_button.click()
    w.table.selectRow(1)
    w.resume_button.click()
    monkeypatch.setattr(jw.QMessageBox, "question", staticmethod(lambda *a, **k: jw.QMessageBox.StandardButton.No))
    w.cancel_button.click()
    assert got == [("pause", "j1"), ("resume", "j2")]                                              # cancelling asks first; "No" does nothing
    monkeypatch.setattr(jw.QMessageBox, "question", staticmethod(lambda *a, **k: jw.QMessageBox.StandardButton.Yes))
    w.cancel_button.click()
    assert got[-1] == ("cancel", "j2")


def test_paused_and_cancelled_jobs_are_described(qapp):
    from doc_explainer_ui.gui.windows.jobs_window import describe
    assert "Paused" in describe(JOBS[1]) and "50%" in describe(JOBS[1])
    assert "Cancelled" in describe(JOBS[3])


# ----------------------------------------------------------------------------- the whole window, wired
def make_window(qapp):
    from doc_explainer_ui.config import UIConfig
    from doc_explainer_ui.gui.models.signals import UISignals
    from doc_explainer_ui.gui.windows.main_window import MainWindow
    calls = []

    class WM:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, *a))
    cap = {}
    wm = WM()
    wm.register_document_async = lambda path, on_done=None, on_error=None, on_progress=None, on_cancelled=None, build_graph=True: cap.update(done=on_done)
    cfg = UIConfig()
    w = MainWindow(wm, SimpleNamespace(toggle_theme=lambda: None), SimpleNamespace(register_action=lambda *a, **k: None), None, UISignals(), cfg)
    return w, calls, cap


def open_ready(w, cap, tmp_path, doc_id="d1"):
    f = tmp_path / "n.txt"
    f.write_text("some text that is long enough to put the cursor well into the document " * 3)
    w.open_document(str(f))
    cap["done"](doc_id)
    return w.tabs.currentWidget()


def put_cursor(viewer, page, position):
    viewer.current_page = page
    cursor = viewer.text_edit.textCursor()
    cursor.setPosition(position)
    viewer.text_edit.setTextCursor(cursor)


def test_typed_and_spoken_questions_are_asked_at_the_readers_position(qapp, tmp_path):
    w, calls, cap = make_window(qapp)
    viewer = open_ready(w, cap, tmp_path)
    put_cursor(viewer, 3, 40)
    chat = w.sidebar.chat_panel
    chat.input.setPlainText("What is this?")
    chat.send_button.click()
    assert ("on_ask", "d1", "What is this?", 3, 40) in calls
    w.voice_input.voice_text.emit("and what about that")                                            # the microphone's words
    assert chat.input.toPlainText() == "and what about that"                                         # land in the box, not sent
    assert not any(c[0] == "on_ask" and c[2] == "and what about that" for c in calls)
    chat.input.clear()
    w.sidebar.recommendations_panel.concept_requested.emit("ATP")
    assert ("on_ask", "d1", "Explain ATP", 3, 40) in calls


def test_i_finished_this_section_uses_the_section_the_reader_is_looking_at(qapp, tmp_path):
    w, calls, cap = make_window(qapp)
    viewer = open_ready(w, cap, tmp_path)
    put_cursor(viewer, 2, 10)
    learn = w.sidebar.learn_panel
    learn.set_knowledge("ready", 3, 5)
    learn.finished_button.click()
    assert ("finish_section", "d1", 2, 10) in calls                                                  # no need to have asked anything first


def test_the_finished_button_without_a_ready_document_explains_itself(qapp):
    w, calls, cap = make_window(qapp)
    w.sidebar.learn_panel.finished_button.setEnabled(True)
    w.sidebar.learn_panel.finished_button.click()
    assert "Open a document" in w.statusBar().currentMessage() and not any(c[0] == "finish_section" for c in calls)


def test_the_microphone_button_starts_and_stops_listening_and_never_without_a_document(qapp, tmp_path, monkeypatch):
    w, calls, cap = make_window(qapp)
    started, stopped = [], []
    monkeypatch.setattr(w.voice_input, "start_listening", lambda: started.append(1))
    monkeypatch.setattr(w.voice_input, "stop_listening", lambda: stopped.append(1))
    chat = w.sidebar.chat_panel
    chat.mic_button.click()                                                                          # no document yet
    assert started == [] and not chat.mic_button.isChecked() and "Open a document" in w.statusBar().currentMessage()
    open_ready(w, cap, tmp_path)
    chat.mic_button.click()
    assert started == [1]
    w.voice_input.recording_changed.emit(False)                                                      # the microphone failed on its own
    assert not chat.mic_button.isChecked()
    chat.mic_button.click()
    chat.mic_button.click()
    assert stopped == [1]


def test_switching_documents_switches_the_microphone_off(qapp, tmp_path, monkeypatch):
    w, calls, cap = make_window(qapp)
    open_ready(w, cap, tmp_path)
    w.voice_input.is_recording = True
    stopped = []
    monkeypatch.setattr(w.voice_input, "stop_listening", lambda: stopped.append(1))
    f = tmp_path / "second.txt"
    f.write_text("x")
    w.open_document(str(f))                                                                          # a second tab becomes current
    assert stopped == [1]


def test_the_chat_session_controls_reach_the_window_manager(qapp, tmp_path):
    w, calls, cap = make_window(qapp)
    open_ready(w, cap, tmp_path)
    chat = w.sidebar.chat_panel
    chat.new_session_requested.emit()
    chat.session_selected.emit("s7")
    chat.note_add_requested.emit("a note")
    chat.summarize_requested.emit()
    chat.stop_requested.emit()
    names = [c[0] for c in calls]
    assert ("new_session", "d1") in calls and ("select_session", "d1", "s7") in calls and ("add_note", "d1", "a note") in calls
    assert ("summarize_session", "d1") in calls and "stop_chat" in names


def test_job_status_buttons_reach_the_window_manager_through_the_real_wiring(qapp):
    from doc_explainer_ui.gui.managers.window_manager import WindowManager
    from doc_explainer_ui.gui.utils.background import Background
    wm = object.__new__(WindowManager)
    wm.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    sent = []
    wm.client = SimpleNamespace(pause_job=lambda j: sent.append(("pause", j)), resume_job=lambda j: sent.append(("resume", j)),
                                cancel_job=lambda j: sent.append(("cancel", j)), list_jobs=lambda: [])
    wm._bg, wm.main_window = Background(), SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *a: None))
    wm.show_jobs()
    wm.jobs_window.pause_requested.emit("j1")
    wm.jobs_window.resume_requested.emit("j1")
    wm.jobs_window.cancel_requested.emit("j2")
    end = time.time() + 5
    while time.time() < end and len(sent) < 3:
        qapp.processEvents()
        time.sleep(0.01)
    assert sent == [("pause", "j1"), ("resume", "j1"), ("cancel", "j2")]
