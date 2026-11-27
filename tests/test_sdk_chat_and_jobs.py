"""The SDK's streaming, sessions, notes and job-control calls against the real app, over BOTH protocols."""
import threading
import time

import pytest

from test_sdk_contract import PROTOCOLS, async_client, make_app, ordinal, sync_client  # noqa: F401  (autouse fixture re-used)
from test_sdk_contract import _close_services  # noqa: F401
from fakes import BOOK, Brain

import doc_explainer_sdk as sdk


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "book.md"
    p.write_text(BOOK, encoding="utf-8")
    return p


def knowledge_started(brain, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if any("build a knowledge graph" in s for s, _ in brain.calls):
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------------- streaming
def test_rest_streams_status_then_deltas_then_done(tmp_path, book):
    c = sync_client(make_app(tmp_path), "rest")
    d = c.register_document(book, poll_interval=0.02).doc_id
    ev = list(c.chat_stream(d, "What is the Calvin cycle?", section_id=3))
    types = [e.type for e in ev]
    assert types[0] == "session" and types[-1] == "done" and "delta" in types
    assert [e.data["stage"] for e in ev if e.type == "status"] == ["understanding", "searching", "thinking", "writing", "finishing"]
    text = "".join(e.text for e in ev if e.type == "delta")
    done = ev[-1].result
    assert text == done.text and done.explanation.follow_up_questions == ["Why?"] and done.session_id == ev[0].data["id"]
    assert sum(e.type == "delta" for e in ev) > 3                                               # really arrives in pieces


def test_rpc_cannot_stream_so_the_same_events_arrive_at_once(tmp_path, book):
    c = sync_client(make_app(tmp_path), "rpc")
    d = c.register_document(book, poll_interval=0.02).doc_id
    ev = list(c.chat_stream(d, "What is ATP?"))
    assert [e.type for e in ev] == ["session", "delta", "done"] and ev[1].text == ev[2].result.text


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_failed_reply_raises_the_matching_error_from_the_iterator(tmp_path, book, protocol):
    brain = Brain()
    c = sync_client(make_app(tmp_path, brain), protocol)
    d = c.register_document(book, poll_interval=0.02).doc_id
    brain.fail_reply = True
    with pytest.raises(sdk.ProcessingError):
        list(c.chat_stream(d, "x"))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_bad_requests_to_the_stream_raise_before_any_event(tmp_path, book, protocol):
    c = sync_client(make_app(tmp_path), protocol)
    d = c.register_document(book, poll_interval=0.02).doc_id
    with pytest.raises(sdk.NotFoundError):
        list(c.chat_stream("nope", "x"))
    with pytest.raises(sdk.NotFoundError) as e:
        list(c.chat_stream(d, "x", session_id="missing"))
    assert e.value.code == "session_not_found"
    with pytest.raises(ValueError):
        list(c.chat_stream(d, "x", kind="dance"))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_async_streaming_and_sessions(tmp_path, book, protocol):
    async with async_client(make_app(tmp_path), protocol) as c:
        doc = await c.register_document(book.read_bytes(), filename="book.md", poll_interval=0.02)
        d = doc.doc_id
        ev = [e async for e in c.chat_stream(d, "What is the Calvin cycle?")]
        assert ev[0].type == "session" and ev[-1].type == "done"
        sid = ev[0].data["id"]
        detail = await c.get_session(d, sid)
        assert [m.role for m in detail.messages] == ["user", "assistant"]
        note = await c.add_note(d, sid, "remember")
        await c.edit_note(d, sid, note.id, "remember this")
        summary = await c.summarize_session(d, sid)
        assert summary.kind == "summary"
        assert [n.text for n in (await c.get_session(d, sid)).notes][0] == "remember this"
        await c.delete_note(d, sid, note.id)
        await c.rename_session(d, sid, "Renamed")
        assert [s.title for s in await c.list_sessions(d)] == ["Renamed"]
        await c.delete_session(d, sid)
        with pytest.raises(sdk.NotFoundError):
            await c.get_session(d, sid)


# ----------------------------------------------------------------------------- sessions and notes
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_sessions_history_notes_and_summaries(tmp_path, book, protocol):
    c = sync_client(make_app(tmp_path), protocol)
    d = c.register_document(book, poll_interval=0.02).doc_id
    a = c.create_session(d, "Exam prep")
    assert a.title == "Exam prep" and c.create_session(d).title == "New session"
    first = c.chat(d, "What is the Calvin cycle?", session_id=a.id)
    assert first.session_id == a.id and first.text.startswith("LEVEL=")
    c.chat(d, "ATP", kind="explain", session_id=a.id)
    c.chat(d, "Light reactions", kind="summarize", session_id=a.id)
    detail = c.get_session(d, a.id)
    assert [m.kind for m in detail.messages] == ["ask", "ask", "explain", "explain", "summarize", "summarize"]
    assert detail.messages[1].meta["follow_ups"] == ["Why?"]
    note = c.add_note(d, a.id, "ATP = energy currency")
    assert note.kind == "note"
    summary = c.summarize_session(d, a.id)
    assert summary.kind == "summary" and "question(s)" in summary.text
    assert [n.kind for n in c.get_session(d, a.id).notes] == ["note", "summary"]
    with pytest.raises(sdk.InvalidRequestError) as e:
        c.summarize_session(d, c.create_session(d, "Empty").id)
    assert e.value.code == "nothing_to_summarize"
    with pytest.raises(sdk.NotFoundError):
        c.edit_note(d, a.id, "nope", "x")


# ----------------------------------------------------------------------------- job control
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_pause_resume_through_the_sdk(tmp_path, book, protocol):
    brain = Brain()
    brain.gate = threading.Event()
    c = sync_client(make_app(tmp_path, brain), protocol)
    job = c.start_registration(book)
    assert knowledge_started(brain)
    assert c.pause_job(job.job_id).stage.startswith("pausing")
    brain.gate.set()
    end = time.time() + 15
    while time.time() < end and c.get_job(job.job_id).state != "paused":
        time.sleep(0.02)
    assert c.get_job(job.job_id).state == "paused" and not c.get_job(job.job_id).done          # paused is a rest stop, not an end
    assert c.resume_job(job.job_id).state in ("queued", "running")
    done = c.wait_for_job(job.job_id, poll_interval=0.02)
    assert done.state == "succeeded" and done.document.concepts == 8


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_cancel_through_the_sdk_ends_waiting_with_a_clear_error(tmp_path, book, protocol):
    brain = Brain()
    brain.gate = threading.Event()
    c = sync_client(make_app(tmp_path, brain), protocol)
    job = c.start_registration(book)
    assert knowledge_started(brain)
    c.cancel_job(job.job_id)
    brain.gate.set()
    with pytest.raises(sdk.JobCancelledError) as e:
        c.wait_for_job(job.job_id, poll_interval=0.02, timeout=15)
    assert e.value.code == "job_cancelled" and "book.md" in e.value.message
    assert c.get_job(job.job_id).done and c.list_documents() == []


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_register_document_raises_when_the_job_is_cancelled_meanwhile(tmp_path, book, protocol):
    brain = Brain()
    brain.gate = threading.Event()
    c = sync_client(make_app(tmp_path, brain), protocol)
    errors = []

    def register():
        try:
            c.register_document(book, poll_interval=0.02)
        except sdk.JobCancelledError as e:
            errors.append(e)
    t = threading.Thread(target=register)
    t.start()
    assert knowledge_started(brain)
    job = c.list_jobs()[0]
    c.cancel_job(job.job_id)
    brain.gate.set()
    t.join(15)
    assert len(errors) == 1


def test_finished_jobs_cannot_be_paused(tmp_path, book):
    c = sync_client(make_app(tmp_path), "rest")
    job = c.start_registration(book)
    c.wait_for_job(job.job_id, poll_interval=0.02)
    with pytest.raises(sdk.InvalidRequestError) as e:
        c.pause_job(job.job_id)
    assert e.value.code == "job_not_active"
