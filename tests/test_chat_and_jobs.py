"""Streamed replies, sessions, notes and summaries, and pausing / resuming / cancelling jobs: over the real HTTP app."""
import json
import threading
import time

import pytest

from fakes import BOOK, Brain
from test_api import _close_services, doc_id, make_client, register, upload, wait_job  # noqa: F401  (fixture re-used)


@pytest.fixture
def brain():
    return Brain()


def events(response):
    """Parse a server-sent-event body into [(event, data)]."""
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return out


def stream(c, d, text, kind="ask", **kw):
    return c.post(f"/v1/documents/{d}/chat/stream", json={"kind": kind, "text": text, **kw})


# ----------------------------------------------------------------------------- streaming
def test_a_reply_streams_status_then_text_then_the_complete_answer(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    r = stream(c, d, "What is the Calvin cycle?", section_id=3)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "session" and kinds[-1] == "done"
    stages = [data["stage"] for k, data in ev if k == "status"]
    assert stages == ["understanding", "searching", "thinking", "writing", "finishing"]       # what the reader is shown, in order
    assert [data["text"] for k, data in ev if k == "status"][0] == "Understanding your request"
    deltas = [data["text"] for k, data in ev if k == "delta"]
    assert len(deltas) > 3 and "===" not in "".join(deltas)                                    # arrives in pieces; no marker
    done = ev[-1][1]
    assert "".join(deltas) == done["text"] and done["explanation"]["follow_up_questions"] == ["Why?"]
    assert kinds.index("delta") > kinds.index("status") and kinds.index("done") > max(i for i, k in enumerate(kinds) if k == "delta")


def test_a_model_failure_arrives_as_an_error_event_and_leaves_the_question_in_the_history(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    brain.fail_reply = True
    ev = events(stream(c, d, "Why?"))
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "processing_failed" and "model" in ev[-1][1]["message"]
    sid = ev[0][1]["id"]
    msgs = c.get(f"/v1/documents/{d}/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user"]                                               # no half answer was stored


def test_bad_requests_are_normal_errors_not_streams(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    assert stream(c, "nope", "x").status_code == 404
    assert stream(c, d, "x", session_id="missing").json()["error"]["code"] == "session_not_found"
    assert c.post(f"/v1/documents/{d}/chat/stream", json={"kind": "dance", "text": "x"}).status_code == 422
    assert c.post(f"/v1/documents/{d}/chat/stream", json={"kind": "ask", "text": ""}).status_code == 422


def test_the_same_turn_without_streaming_over_rest_and_rpc(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    r = c.post(f"/v1/documents/{d}/chat", json={"kind": "explain", "text": "ATP", "section_id": 3})
    assert r.status_code == 200 and r.json()["text"].startswith("LEVEL=") and r.json()["session_id"]
    rpc = c.post("/rpc", json={"jsonrpc": "2.0", "id": 1, "method": "chat.send",
                               "params": {"doc_id": d, "kind": "ask", "text": "What?", "session_id": r.json()["session_id"]}}).json()
    assert rpc["result"]["session_id"] == r.json()["session_id"] and rpc["result"]["text"].startswith("LEVEL=")


# ----------------------------------------------------------------------------- sessions, history
def test_history_keeps_both_sides_with_follow_ups_and_names_the_session_after_the_first_question(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    first = events(stream(c, d, "What is the Calvin cycle?", section_id=3))
    sid = first[0][1]["id"]
    second = events(stream(c, d, "And ATP?", section_id=3))
    assert second[0][1]["id"] == sid                                                           # same session by default
    detail = c.get(f"/v1/documents/{d}/sessions/{sid}").json()
    assert detail["session"]["title"] == "What is the Calvin cycle?" and detail["session"]["messages"] == 4
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][0]["text"] == "What is the Calvin cycle?" and detail["messages"][1]["meta"]["follow_ups"] == ["Why?"]
    assert detail["messages"][1]["meta"]["resources"] is not None


def test_sessions_can_be_created_listed_renamed_and_deleted(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    a = c.post(f"/v1/documents/{d}/sessions", json={"title": "Exam prep"}).json()
    b = c.post(f"/v1/documents/{d}/sessions", json={}).json()
    assert a["title"] == "Exam prep" and b["title"] == "New session"
    events(stream(c, d, "hello", session_id=a["id"]))                                          # a is now the most recent
    assert [s["id"] for s in c.get(f"/v1/documents/{d}/sessions").json()["sessions"]][0] == a["id"]
    renamed = c.patch(f"/v1/documents/{d}/sessions/{b['id']}", json={"title": "Chapter 2"}).json()
    assert renamed["title"] == "Chapter 2"
    assert c.delete(f"/v1/documents/{d}/sessions/{a['id']}").status_code == 204
    assert c.get(f"/v1/documents/{d}/sessions/{a['id']}").json()["error"]["code"] == "session_not_found"
    assert [s["id"] for s in c.get(f"/v1/documents/{d}/sessions").json()["sessions"]] == [b["id"]]


def test_a_fresh_session_is_not_polluted_by_another_and_the_two_stay_separate(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    one = events(stream(c, d, "first topic"))[0][1]["id"]
    two = c.post(f"/v1/documents/{d}/sessions", json={"title": "Second"}).json()["id"]
    events(stream(c, d, "second topic", session_id=two))
    assert [m["text"] for m in c.get(f"/v1/documents/{d}/sessions/{one}").json()["messages"]][0] == "first topic"
    assert len(c.get(f"/v1/documents/{d}/sessions/{two}").json()["messages"]) == 2


def test_conversations_are_private_to_their_user(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c, headers={"X-User-Id": "alice"})
    sid = events(stream_as(c, d, "alice's secret question", "alice"))[0][1]["id"]
    assert c.get(f"/v1/documents/{d}/sessions/{sid}", headers={"X-User-Id": "bob"}).status_code == 404
    assert c.get(f"/v1/documents/{d}/sessions", headers={"X-User-Id": "bob"}).status_code == 404       # not even the document
    assert c.get(f"/v1/documents/{d}/sessions/{sid}", headers={"X-User-Id": "alice"}).status_code == 200


def stream_as(c, d, text, user):
    return c.post(f"/v1/documents/{d}/chat/stream", json={"kind": "ask", "text": text}, headers={"X-User-Id": user})


def test_deleting_the_document_deletes_its_conversations(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    sid = events(stream(c, d, "hello"))[0][1]["id"]
    service = c.app.state.service
    assert service._sys.chat_store.messages(sid)
    c.delete(f"/v1/documents/{d}")
    assert service._sys.chat_store.messages(sid) == [] and service._sys.chat_store.list_sessions("default", d) == []


# ----------------------------------------------------------------------------- notes and summaries
def test_notes_can_be_added_edited_and_deleted(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    sid = c.post(f"/v1/documents/{d}/sessions", json={}).json()["id"]
    n = c.post(f"/v1/documents/{d}/sessions/{sid}/notes", json={"text": "Remember: ATP is the energy currency"}).json()
    assert n["kind"] == "note"
    assert c.put(f"/v1/documents/{d}/sessions/{sid}/notes/{n['id']}", json={"text": "Edited"}).status_code == 204
    detail = c.get(f"/v1/documents/{d}/sessions/{sid}").json()
    assert [x["text"] for x in detail["notes"]] == ["Edited"] and detail["session"]["notes"] == 1
    assert c.post(f"/v1/documents/{d}/sessions/{sid}/notes", json={"text": "  "}).status_code == 422
    assert c.delete(f"/v1/documents/{d}/sessions/{sid}/notes/{n['id']}").status_code == 204
    assert c.delete(f"/v1/documents/{d}/sessions/{sid}/notes/{n['id']}").json()["error"]["code"] == "note_not_found"


def test_the_tutor_can_summarise_a_conversation_into_a_note(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    sid = events(stream(c, d, "What is the Calvin cycle?"))[0][1]["id"]
    events(stream(c, d, "And ATP?"))
    note = c.post(f"/v1/documents/{d}/sessions/{sid}/summary").json()
    assert note["kind"] == "summary" and "2 question(s) asked" in note["text"]
    notes = c.get(f"/v1/documents/{d}/sessions/{sid}").json()["notes"]
    assert [n["kind"] for n in notes] == ["summary"]
    sent = [u for s, u in brain.calls if "study notes" in s][0]
    assert "READER: What is the Calvin cycle?" in sent and "TUTOR:" in sent                       # built from the conversation only


def test_there_is_nothing_to_summarise_before_the_first_answer(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    sid = c.post(f"/v1/documents/{d}/sessions", json={}).json()["id"]
    r = c.post(f"/v1/documents/{d}/sessions/{sid}/summary")
    assert r.status_code == 409 and r.json()["error"]["code"] == "nothing_to_summarize"


def test_sessions_and_notes_over_json_rpc(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)

    def rpc(method, **params):
        return c.post("/rpc", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).json()
    sid = rpc("session.create", doc_id=d, title="T")["result"]["id"]
    assert rpc("session.list", doc_id=d)["result"]["sessions"][0]["id"] == sid
    n = rpc("note.add", doc_id=d, session_id=sid, text="hi")["result"]
    assert rpc("note.edit", doc_id=d, session_id=sid, note_id=n["id"], text="ho")["result"] == {"ok": True}
    assert rpc("session.get", doc_id=d, session_id=sid)["result"]["notes"][0]["text"] == "ho"
    assert rpc("session.rename", doc_id=d, session_id=sid, title="U")["result"]["title"] == "U"
    assert rpc("note.delete", doc_id=d, session_id=sid, note_id=n["id"])["result"] == {"ok": True}
    assert rpc("session.delete", doc_id=d, session_id=sid)["result"] == {"ok": True}
    assert rpc("session.get", doc_id=d, session_id=sid)["error"]["data"]["code"] == "session_not_found"


def test_the_plain_explain_call_records_into_a_session_only_when_asked(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    c.post(f"/v1/documents/{d}/ask", json={"question": "no session"})
    assert c.get(f"/v1/documents/{d}/sessions").json()["sessions"] == []                         # nothing was recorded silently
    sid = c.post(f"/v1/documents/{d}/sessions", json={}).json()["id"]
    c.post(f"/v1/documents/{d}/explain", json={"selected_text": "ATP", "session_id": sid})
    assert [m["kind"] for m in c.get(f"/v1/documents/{d}/sessions/{sid}").json()["messages"]] == ["explain", "explain"]


# ----------------------------------------------------------------------------- pause / resume / cancel
def knowledge_started(brain, n=1, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if sum("build a knowledge graph" in s for s, _ in brain.calls) >= n:
            return True
        time.sleep(0.01)
    return False


def wait_state(c, job_id, *states, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {states}: {job}")


def post(c, job, action):
    return c.post(f"/v1/jobs/{job}/{action}")


def knowledge_calls(brain):
    return sum("build a knowledge graph" in s for s, _ in brain.calls)


def test_pause_stops_at_the_next_step_keeps_the_work_and_resume_finishes_it(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()["job_id"]
    assert knowledge_started(brain)
    r = post(c, job, "pause")
    assert r.status_code == 200 and "pausing" in r.json()["stage"]
    brain.gate.set()
    paused = wait_state(c, job, "paused")
    assert paused["stage"] == "paused" and paused["document"] is None
    assert c.get("/v1/documents").json()["documents"] == []                                      # not registered until it finishes
    done_before = knowledge_calls(brain)
    assert done_before < 4                                                                        # it really stopped early
    resumed = post(c, job, "resume").json()
    assert resumed["state"] in ("queued", "running")
    final = wait_job(c, job)
    assert final["state"] == "succeeded" and final["document"]["knowledge"] == "ready" and final["document"]["concepts"] == 8
    assert knowledge_calls(brain) == 4                                                            # finished sections were NOT redone


def test_cancel_a_running_job_throws_everything_away(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()["job_id"]
    assert knowledge_started(brain)
    assert "cancelling" in post(c, job, "cancel").json()["stage"]
    brain.gate.set()
    final = wait_state(c, job, "cancelled")
    assert final["stage"] == "cancelled" and final["finished_at"] and final["document"] is None
    assert c.get("/v1/documents").json()["documents"] == []
    service = c.app.state.service
    assert service._sys.vectors.count(service._jobs.peek(job)["doc_id"]) == 0                   # the index was cleaned up too
    assert not list(service._uploads.glob("*/*"))                                                  # and the upload


def test_a_queued_job_can_be_paused_it_is_skipped_and_resumed_later(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    first = upload(c, "a.md", (BOOK + "\n\nfirst\n").encode()).json()["job_id"]
    second = upload(c, "b.md", (BOOK + "\n\nsecond\n").encode()).json()["job_id"]
    assert knowledge_started(brain)
    assert c.get(f"/v1/jobs/{second}").json()["state"] == "queued"
    assert post(c, second, "pause").json()["state"] == "paused"
    brain.gate.set()
    assert wait_job(c, first)["state"] == "succeeded"
    time.sleep(0.3)
    assert c.get(f"/v1/jobs/{second}").json()["state"] == "paused"                               # the worker walked past it
    assert len(c.get("/v1/documents").json()["documents"]) == 1
    post(c, second, "resume")
    assert wait_job(c, second)["state"] == "succeeded" and len(c.get("/v1/documents").json()["documents"]) == 2


def test_a_paused_job_can_be_cancelled(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()["job_id"]
    assert knowledge_started(brain)
    post(c, job, "pause")
    brain.gate.set()
    wait_state(c, job, "paused")
    assert post(c, job, "cancel").json()["state"] == "cancelled"
    assert c.get("/v1/documents").json()["documents"] == []


def test_resume_before_a_pending_pause_has_taken_effect_just_carries_on(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()["job_id"]
    assert knowledge_started(brain)
    post(c, job, "pause")
    assert post(c, job, "resume").json()["stage"] == "resuming"                                  # changed their mind in time
    brain.gate.set()
    assert wait_job(c, job)["state"] == "succeeded"


def test_job_control_is_idempotent_and_refuses_finished_jobs(tmp_path, brain):
    c = make_client(tmp_path, brain)
    done = register(c)
    r = post(c, done["job_id"], "pause")
    assert r.status_code == 409 and r.json()["error"]["code"] == "job_not_active"
    assert post(c, done["job_id"], "cancel").json()["error"]["code"] == "job_not_active"
    assert post(c, done["job_id"], "resume").json()["error"]["code"] == "job_not_active"
    brain.gate = threading.Event()
    job = upload(c, "z.md", (BOOK + "\n\nz\n").encode()).json()["job_id"]
    post(c, job, "pause")
    brain.gate.set()
    wait_state(c, job, "paused")
    assert post(c, job, "pause").json()["state"] == "paused"                                     # pausing twice is fine
    assert post(c, job, "cancel").json()["state"] == "cancelled" and post(c, job, "cancel").json()["state"] == "cancelled"


def test_job_control_belongs_to_the_owner_and_works_over_rpc(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = c.post("/v1/documents", files={"file": ("a.md", BOOK.encode())}, headers={"X-User-Id": "alice"}).json()["job_id"]
    assert knowledge_started(brain)
    assert c.post(f"/v1/jobs/{job}/cancel", headers={"X-User-Id": "bob"}).status_code == 404
    out = c.post("/rpc", json={"jsonrpc": "2.0", "id": 1, "method": "job.pause", "params": {"job_id": job}},
                 headers={"X-User-Id": "alice"}).json()
    assert out["result"]["stage"].startswith("pausing")
    brain.gate.set()
    end = time.time() + 10
    while time.time() < end and c.get(f"/v1/jobs/{job}", headers={"X-User-Id": "alice"}).json()["state"] != "paused":
        time.sleep(0.02)
    for method, state in (("job.resume", "queued"), ("job.cancel", None)):
        res = c.post("/rpc", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": {"job_id": job}}, headers={"X-User-Id": "alice"}).json()
        assert "result" in res


def test_cancelling_a_retry_leaves_the_existing_document_alone(tmp_path, brain):
    brain.fail_knowledge = True
    c = make_client(tmp_path, brain)
    d = register(c)["document"]["doc_id"]                                                         # registered, knowledge failed
    brain.fail_knowledge = False
    brain.gate = threading.Event()
    retry = upload(c).json()["job_id"]
    assert knowledge_started(brain, 4)
    post(c, retry, "cancel")
    brain.gate.set()
    assert wait_state(c, retry, "cancelled")["state"] == "cancelled"
    assert c.get(f"/v1/documents/{d}").status_code == 200 and c.get(f"/v1/documents/{d}/structure").status_code == 200


def test_a_paused_job_survives_a_server_restart_and_can_be_resumed(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()["job_id"]
    assert knowledge_started(brain)
    post(c, job, "pause")
    brain.gate.set()
    wait_state(c, job, "paused")
    c.app.state.service.close()
    brain2 = Brain()
    c2 = make_client(tmp_path, brain2)                                                            # a new server over the same folders
    assert c2.get(f"/v1/jobs/{job}").json()["state"] == "paused"                                  # not marked failed
    post(c2, job, "resume")
    final = wait_job(c2, job)
    assert final["state"] == "succeeded" and final["document"]["concepts"] == 8
