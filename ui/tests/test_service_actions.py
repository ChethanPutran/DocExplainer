"""WindowManager's service calls, with the REAL SDK talking to a mock server."""
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from doc_explainer_sdk import DocExplainerClient
from doc_explainer_ui.gui.managers.window_manager import WindowManager
from doc_explainer_ui.gui.utils.background import Background
from doc_explainer_ui.gui.widgets import Sidebar

DOC = {"doc_id": "d1", "title": "T", "filename": "f.pdf", "size_bytes": 3, "uploaded_at": "now"}
RESULT = {"text": "Because it is", "explanation": {"explanation": "Because it is", "follow_up_questions": ["why?"],
                                                    "context_used": {"concept_ids": ["c1"]}, "resources": []},
          "session_id": "s1", "message_id": 2}


def sse(*events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def good_stream():
    return sse(("session", {"id": "s1"}), ("status", {"stage": "understanding", "text": "Understanding your request"}),
               ("status", {"stage": "writing", "text": "Writing the reply"}), ("delta", {"text": "Because "}),
               ("delta", {"text": "it is"}), ("done", RESULT))


def job(state, **kw):
    base = {"job_id": "j1", "state": state, "filename": "f.pdf", "created_at": "now"}
    if state == "succeeded":
        base["document"] = DOC
    return {**base, **kw}


class FakeMainWindow:
    def __init__(self):
        self.messages = []
        self.sidebar = Sidebar()
        self.panel = self.sidebar.chat_panel

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda m, ms=0: self.messages.append(m))


def make_wm(handler, tmp_path):
    wm = object.__new__(WindowManager)                  # skip the Qt-heavy constructor
    wm.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    wm.client = DocExplainerClient("http://srv", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    wm._bg, wm._registered, wm._structures, wm.main_window = Background(), set(), {}, FakeMainWindow()
    wm._sessions, wm._chatting, wm._chat_stop = {}, False, None
    f = tmp_path / "f.pdf"
    f.write_bytes(b"%PDF-1.4 x")
    return wm, str(f)


class Server:
    """Mock service: walks a job through `states`, answers explain/section, counts requests."""

    def __init__(self, states, explain=None):
        self.states, self.polls, self.requests, self.explain, self.bodies = list(states), 0, [], explain, []

    def __call__(self, req):
        self.requests.append((req.method, req.url.path))
        self.bodies.append((req.method, req.url.path, req.content.decode() if req.content else ""))
        path = req.url.path
        if req.method == "POST" and path == "/v1/documents":
            return httpx.Response(202, json=job("queued", queue_position=2))
        if path.startswith("/v1/jobs/"):
            s = self.states[min(self.polls, len(self.states) - 1)]
            self.polls += 1
            return httpx.Response(200, json=s)
        if path.endswith("/section"):
            return httpx.Response(200, json={"section_id": 7})
        if path.endswith("/chat/stream"):
            if self.explain:
                return self.explain(req)
            return httpx.Response(200, content=good_stream(), headers={"content-type": "text/event-stream"})
        if path.endswith("/sessions"):
            return httpx.Response(200, json={"sessions": []})
        return httpx.Response(200, json={})


def test_registration_does_not_block_and_reports_progress(tmp_path, pump):
    server = Server([job("queued", queue_position=1), job("running"), job("running"), job("succeeded")])
    wm, path = make_wm(server, tmp_path)
    done, threads = [], []
    wm.register_document_async(path, on_done=lambda d: (done.append(d), threads.append(threading.current_thread())))
    assert done == []                                         # returned immediately
    assert pump(lambda: done == ["d1"])
    assert threads == [threading.main_thread()] and "d1" in wm._registered
    shown = " | ".join(wm.main_window.messages)
    assert "position 2" in shown and "Processing the document" in shown and "Document ready" in shown


def test_failed_registration_reports_the_real_reason(tmp_path, pump):
    failed = job("failed", error={"code": "processing_failed", "message": "cannot parse"})
    wm, path = make_wm(Server([job("running"), failed]), tmp_path)
    errors, done = [], []
    wm.register_document_async(path, on_done=done.append, on_error=errors.append)
    assert pump(lambda: errors)
    assert errors == ["cannot parse"] and done == [] and wm._registered == set()
    assert "cannot parse" in wm.main_window.messages[-1]


def test_unreachable_server_is_reported_not_raised(tmp_path, pump):
    def refuse(req):
        raise httpx.ConnectError("refused")
    wm, path = make_wm(refuse, tmp_path)
    errors = []
    wm.register_document_async(path, on_error=errors.append)
    assert pump(lambda: errors) and "Cannot reach" in errors[0]


def test_actions_wait_until_the_document_is_ready(tmp_path, pump):
    server = Server([job("succeeded")])
    wm, _ = make_wm(server, tmp_path)
    wm.on_explain("d1", "text", 1, 2)                         # not registered yet
    wm.on_ask("d1", "q", 1, 2)
    assert "still being processed" in wm.main_window.messages[-1]
    time.sleep(0.1)
    assert server.requests == []                              # nothing was sent to the service


def test_a_question_streams_into_the_chat_for_the_located_section(tmp_path, pump):
    server = Server([job("succeeded")])
    wm, _ = make_wm(server, tmp_path)
    wm._registered.add("d1")
    panel = wm.main_window.panel
    wm.on_ask("d1", "Why?", 3, 4)
    assert pump(lambda: [m.role for m in panel.messages()] == ["user", "assistant"] and not panel._busy)
    assert [m.text for m in panel.messages()] == ["Why?", "Because it is"]
    body = json.loads(next(r for r in server.bodies if r[1].endswith("/chat/stream"))[2])
    assert body == {"kind": "ask", "text": "Why?", "section_id": 7}                     # the section was located first
    assert wm._sessions["d1"] == "s1" and not wm._chatting
    assert [b.text() for b in panel.messages()[1].follow_up_buttons() if b.text() == "why?"] == ["why?"]
    assert wm.main_window.sidebar._concept_ids == ["c1"]                                # feedback can now be given


def test_explain_and_summarize_say_what_was_asked_and_use_the_right_kind(tmp_path, pump):
    server = Server([job("succeeded")])
    wm, _ = make_wm(server, tmp_path)
    wm._registered.add("d1")
    wm.on_explain("d1", "some selected text", 1, 1)
    assert pump(lambda: not wm._chatting and len(wm.main_window.panel.messages()) == 2)
    wm.on_summarize("d1", "other text", 1, 1)
    assert pump(lambda: not wm._chatting and len(wm.main_window.panel.messages()) == 4)
    texts = [m.text for m in wm.main_window.panel.messages()]
    assert texts[0].startswith("Explain: \u201csome selected text") and texts[2].startswith("Summarize: \u201cother text")
    kinds = [json.loads(r[2])["kind"] for r in server.bodies if r[1].endswith("/chat/stream")]
    assert kinds == ["explain", "summarize"]
    assert [json.loads(r[2]).get("session_id") for r in server.bodies if r[1].endswith("/chat/stream")] == [None, "s1"]   # same session


def test_the_second_question_waits_for_the_first(tmp_path, pump):
    gate = threading.Event()

    def slow(req):
        gate.wait(5)
        return httpx.Response(200, content=good_stream(), headers={"content-type": "text/event-stream"})
    wm, _ = make_wm(Server([job("succeeded")], explain=slow), tmp_path)
    wm._registered.add("d1")
    wm.on_ask("d1", "first", 1, 1)
    wm.on_ask("d1", "second", 1, 1)
    assert "still answering" in wm.main_window.messages[-1]
    assert [m.text for m in wm.main_window.panel.messages()][:1] == ["first"] and len(wm.main_window.panel.messages()) == 2
    gate.set()
    assert pump(lambda: not wm._chatting)


def test_stop_keeps_what_was_written_and_frees_the_chat(tmp_path, pump):
    gate = threading.Event()

    def slow(req):
        gate.wait(5)
        return httpx.Response(200, content=good_stream(), headers={"content-type": "text/event-stream"})
    wm, _ = make_wm(Server([job("succeeded")], explain=slow), tmp_path)
    wm._registered.add("d1")
    wm.on_ask("d1", "first", 1, 1)
    wm.main_window.panel.append_delta("Partial answer")
    wm.stop_chat()
    assert not wm._chatting and not wm.main_window.panel._busy
    assert "Partial answer" in wm.main_window.panel.messages()[1].text and "stopped" in wm.main_window.panel.messages()[1].text
    gate.set()
    wm.on_ask("d1", "next", 1, 1)                                   # usable again at once
    assert pump(lambda: len(wm.main_window.panel.messages()) == 4)


def test_a_busy_server_is_shown_in_the_chat_and_nothing_crashes(tmp_path, pump):
    busy = lambda req: httpx.Response(503, json={"error": {"code": "not_ready", "message": "The server is still starting up."}})
    wm, _ = make_wm(Server([job("succeeded")], explain=busy), tmp_path)
    wm._registered.add("d1")
    wm.on_ask("d1", "q", 1, 1)
    assert pump(lambda: any("still starting up" in m.text for m in wm.main_window.panel.messages()))
    assert not wm._chatting and not wm.main_window.panel._busy


def test_a_mid_stream_error_event_is_shown_in_place_of_the_answer(tmp_path, pump):
    broken = lambda req: httpx.Response(200, content=sse(("session", {"id": "s1"}), ("delta", {"text": "Half"}),
                                                          ("error", {"code": "processing_failed", "message": "the model is down"})),
                                        headers={"content-type": "text/event-stream"})
    wm, _ = make_wm(Server([job("succeeded")], explain=broken), tmp_path)
    wm._registered.add("d1")
    wm.on_ask("d1", "q", 1, 1)
    assert pump(lambda: not wm._chatting and "the model is down" in wm.main_window.panel.messages()[1].text)
    assert "Half" in wm.main_window.panel.messages()[1].text


def test_closing_the_app_cancels_a_long_wait(tmp_path, pump):
    wm, path = make_wm(Server([job("running")]), tmp_path)    # never finishes
    errors = []
    wm.register_document_async(path, on_error=errors.append)
    assert pump(lambda: wm.main_window.messages)              # the wait is under way
    wm._bg.shutdown()
    assert pump(lambda: wm._bg.active == 0, timeout=8)        # stops within one 2s polling slice
    assert errors == []                                       # and shows no error dialog
