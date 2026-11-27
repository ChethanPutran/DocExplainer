"""Contract test: the SDK against the real FastAPI app and the real system, over BOTH protocols, sync and async.

Only the language model is scripted. This is what keeps the SDK's own copy of the wire models and its two
protocol implementations in step with the server.
"""
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

sdk = pytest.importorskip("doc_explainer_sdk")
from doc_explainer.api import ApiSettings, create_app  # noqa: E402
from doc_explainer.api.service import DocumentApiService  # noqa: E402
from doc_explainer.orchestrator import System, SystemConfig  # noqa: E402

from fakes import BOOK, Brain, HashingEmbedder  # noqa: E402

PROTOCOLS = ["rest", "rpc"]
BASE = "http://testserver"
_services = []


@pytest.fixture(autouse=True)
def _close_services():
    yield
    while _services:
        _services.pop().close()


def make_app(tmp_path, brain=None, **kw):
    settings = ApiSettings(data_dir=tmp_path / "api", **kw)
    system = System(SystemConfig(data_dir=tmp_path / "system"), llm=brain or Brain(), embedder=HashingEmbedder())
    service = DocumentApiService(system, settings)
    _services.append(service)
    return create_app(service=service, settings=settings)


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "book.md"
    p.write_text(BOOK, encoding="utf-8")
    return p


def sync_client(app, protocol, **kw):
    return sdk.DocExplainerClient(BASE, protocol=protocol, http_client=TestClient(app), **kw)


def async_client(app, protocol, **kw):
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)
    return sdk.AsyncDocExplainerClient(BASE, protocol=protocol, http_client=http, **kw)


def ordinal(structure, title):
    return next(s.ordinal for s in structure.sections if s.title == title)


# ----------------------------------------------------------------------------- everything, over each protocol
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_sync_full_flow(tmp_path, book, protocol):
    c = sync_client(make_app(tmp_path), protocol)
    h = c.health()
    assert h.status == "ok" and h.protocols == ["rest", "rpc"] and ".md" in h.supported_types
    doc = c.register_document(book, poll_interval=0.02)
    assert (doc.title, doc.filename, doc.knowledge, doc.sections, doc.concepts) == ("Photosynthesis", "book.md", "ready", 4, 8)
    assert [d.doc_id for d in c.list_documents()] == [doc.doc_id] and c.get_document(doc.doc_id).title == "Photosynthesis"
    d = doc.doc_id

    s = c.get_structure(d)
    assert [(x.title, x.depth) for x in s.sections] == [("Photosynthesis", 1), ("Light reactions", 2), ("Calvin cycle", 2), ("Cellular respiration", 2)]
    calvin = ordinal(s, "Calvin cycle")
    assert c.get_section_id(d, 1, 0) in {x.ordinal for x in s.sections}

    r = c.explain(d, "The Calvin cycle uses ATP.", section_id=calvin)
    assert r.text.startswith("LEVEL=") and r.explanation.context_used["section"] == "Photosynthesis > Calvin cycle"
    assert r.explanation.style.level == "intermediate" and r.explanation.follow_up_questions == ["Why?"]
    assert c.explain(d, "ATP", section_id=calvin, style=sdk.ExplanationStyle(level="expert", depth="fixed")).text.startswith("LEVEL=expert")
    assert c.explain(d, "ATP", section_id=calvin, style={"level": sdk.ExplanationLevel.BEGINNER, "depth": "fixed"}).text.startswith("LEVEL=beginner")
    assert c.summarize(d, "x", section_id=calvin).text and c.ask(d, "What is ATP?", section_id=calvin).text
    ctx = c.get_context(d, calvin)
    assert ctx.document_context["introduces"] and "ATP" in ctx.user_knowledge

    k = c.get_knowledge(d)
    assert len(k.concepts) == 8 and k.as_of is None and any(x.type == "prerequisite_of" for x in k.relations)
    assert 0 < len(c.get_knowledge(d, upto=ordinal(s, "Light reactions") + 5).concepts) < 8
    sk = c.get_section_knowledge(d, calvin)
    assert {x.name for x in sk.introduces} == {"Calvin cycle", "carbon dioxide", "sugar"} and [x.title for x in sk.depends_on] == ["Light reactions"]
    assert [p.total_concepts for p in c.get_growth(d)] == [2, 4, 7, 8]
    assert [(x["name"], x["distance"]) for x in c.get_prerequisites(d, "cellular respiration").data] == [("sugar", 1), ("carbon dioxide", 2)]
    assert [x["name"] for x in c.get_learning_path(d, "cellular respiration").data][-1] == "cellular respiration"
    assert c.recommend_concepts(d, "photosynthesis").data

    diag = c.start_diagnostic(d, "beginner", known_topics=["photosynthesis"], questions=3)
    assert len(diag.questions) == 3 and diag.level == "beginner"
    assert c.explain(d, "The Calvin cycle uses ATP.", section_id=calvin).text.startswith("LEVEL=beginner")
    first = c.answer_quiz(d, diag.questions[0].id, 0)
    assert set(first.model_dump()) == {"correct", "correct_index", "explanation", "already_answered"} and not first.already_answered
    assert c.answer_quiz(d, diag.questions[0].id, 1).already_answered
    assert c.finish_diagnostic(d).questions == []
    learner = c.get_learner(d)
    assert learner.concepts and all(0 <= b.p_known <= 1 for b in learner.concepts)
    quiz = c.practice_quiz(d, section_id=calvin, count=2)
    assert 1 <= len(quiz) <= 2 and quiz[0].kind == "practice"
    c.send_feedback(d, "too_hard", [learner.concepts[0].concept_id])
    c.mark_progress(d, calvin)
    assert c.get_learner(d).position is not None
    c.delete_document(d)
    with pytest.raises(sdk.NotFoundError):
        c.get_document(d)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_async_full_flow(tmp_path, book, protocol):
    async with async_client(make_app(tmp_path), protocol) as c:
        doc = await c.register_document(book.read_bytes(), filename="book.md", poll_interval=0.02)
        d = doc.doc_id
        s = await c.get_structure(d)
        calvin = ordinal(s, "Calvin cycle")
        assert (await c.explain(d, "ATP", section_id=calvin)).text.startswith("LEVEL=")
        assert len((await c.get_knowledge(d)).concepts) == 8 and len(await c.get_growth(d)) == 4
        assert (await c.get_prerequisites(d, "cellular respiration")).data[0]["name"] == "sugar"
        diag = await c.start_diagnostic(d, "beginner", questions=2)
        assert len(diag.questions) == 2 and (await c.answer_quiz(d, diag.questions[0].id, 0)).explanation == "because"
        assert (await c.finish_diagnostic(d)).questions == [] and (await c.get_learner(d)).concepts
        assert len(await c.practice_quiz(d, count=1)) == 1
        await c.send_feedback(d, "too_easy", [(await c.get_learner(d)).concepts[0].concept_id])
        await c.mark_progress(d, calvin)
        await c.delete_document(d)


def test_rest_and_rpc_see_exactly_the_same_system(tmp_path, book):
    app = make_app(tmp_path)
    rest, rpc_ = sync_client(app, "rest"), sync_client(app, "rpc")
    d = rest.register_document(book, poll_interval=0.02).doc_id
    calvin = ordinal(rest.get_structure(d), "Calvin cycle")
    for call in (lambda c: c.get_structure(d), lambda c: c.get_knowledge(d), lambda c: c.get_growth(d),
                 lambda c: c.get_section_knowledge(d, calvin), lambda c: c.get_prerequisites(d, "cellular respiration"),
                 lambda c: c.get_learning_path(d, "cellular respiration"), lambda c: c.get_document(d)):
        assert call(rest) == call(rpc_)
    a, b = (c.explain(d, "The Calvin cycle uses ATP.", section_id=calvin) for c in (rest, rpc_))
    for x in (a, b):
        x.explanation.metadata.processing_time_ms = 0
    assert a == b


# ----------------------------------------------------------------------------- background jobs, progress
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_registration_progress_is_visible_from_the_sdk(tmp_path, book, protocol):
    brain = Brain()
    brain.gate = threading.Event()
    c = sync_client(make_app(tmp_path, brain), protocol)
    job = c.start_registration(book)
    assert not job.done
    deadline = time.time() + 10
    while time.time() < deadline and not any("build a knowledge graph" in s for s, _ in brain.calls):
        time.sleep(0.02)
    mid = c.get_job(job.job_id)
    assert mid.state == "running" and 0.35 <= mid.progress < 1.0 and mid.stage.startswith("reading")
    seen = []
    threading.Timer(0.2, brain.gate.set).start()
    done = c.wait_for_job(job.job_id, poll_interval=0.02, on_update=lambda j: seen.append((j.state, j.progress)))
    assert done.state == "succeeded" and done.progress == 1.0 and done.document.knowledge == "ready"
    assert seen[-1][0] == "succeeded"


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_corrupt_file_raises_ProcessingError_naming_the_file(tmp_path, protocol):
    c = sync_client(make_app(tmp_path), protocol)
    with pytest.raises(sdk.ProcessingError) as e:
        c.register_document(b"definitely not a pdf", filename="broken.pdf", poll_interval=0.02)
    assert e.value.code == "processing_failed" and "broken.pdf" in e.value.message


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_uploading_the_same_file_again_is_instant_and_not_reprocessed(tmp_path, book, protocol):
    brain = Brain()
    c = sync_client(make_app(tmp_path, brain), protocol)
    first = c.register_document(book, poll_interval=0.02)
    calls = len(brain.calls)
    job = c.start_registration(book)
    assert job.done and job.reused and job.document.doc_id == first.doc_id and len(brain.calls) == calls


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_full_queue_surfaces_as_a_retryable_error(tmp_path, protocol):
    brain = Brain()
    brain.gate = threading.Event()
    c = sync_client(make_app(tmp_path, brain, max_jobs_per_user=1), protocol)
    c.start_registration((BOOK + "\n\nA\n").encode(), filename="a.md")
    with pytest.raises(sdk.ServerBusyError) as e:
        c.start_registration((BOOK + "\n\nB\n").encode(), filename="b.md")
    assert e.value.code == "queue_full" and isinstance(e.value, sdk.ServerUnavailableError)
    brain.gate.set()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_questions_are_answered_while_another_document_is_still_processing(tmp_path, book, protocol):
    brain = Brain()
    c = sync_client(make_app(tmp_path, brain), protocol)
    d = c.register_document(book, poll_interval=0.02).doc_id
    brain.gate = threading.Event()
    other = c.start_registration(b"# Rocks\n\nRocks are made of minerals.\n\n## More\n\nMore about geology.\n", filename="r.md")
    deadline = time.time() + 10
    while time.time() < deadline and not any("geology" in u for _, u in brain.calls):
        time.sleep(0.02)
    t0 = time.time()
    assert c.ask(d, "What is the Calvin cycle?").text.startswith("LEVEL=") and time.time() - t0 < 3
    brain.gate.set()
    c.wait_for_job(other.job_id, poll_interval=0.02)


# ----------------------------------------------------------------------------- errors, auth, isolation
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_errors_map_to_the_same_exceptions(tmp_path, book, protocol):
    c = sync_client(make_app(tmp_path), protocol)
    d = c.register_document(book, poll_interval=0.02).doc_id
    with pytest.raises(sdk.NotFoundError) as e:
        c.get_document("nope")
    assert e.value.code == "document_not_found"
    with pytest.raises(sdk.NotFoundError) as e:
        c.get_prerequisites(d, "unobtainium")
    assert e.value.code == "concept_not_found"
    with pytest.raises(sdk.NotFoundError) as e:
        c.answer_quiz(d, "no-such-question", 0)
    assert e.value.code == "question_not_found"
    with pytest.raises(sdk.InvalidRequestError) as e:
        c.register_document(b"x", filename="virus.exe")
    assert e.value.code == "unsupported_file_type"
    with pytest.raises(sdk.InvalidRequestError) as e:
        c.explain(d, "")
    assert e.value.code == "invalid_params"
    with pytest.raises(sdk.InvalidRequestError):
        c.get_section_knowledge(d, 3)                                                       # not a section's position


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_api_key_and_user_isolation(tmp_path, book, protocol):
    app = make_app(tmp_path, api_key="secret")
    with pytest.raises(sdk.AuthenticationError):
        sync_client(app, protocol).list_documents()
    with pytest.raises(sdk.AuthenticationError):
        sync_client(app, protocol, api_key="wrong").list_documents()
    if protocol == "rest":
        assert sync_client(app, protocol).health().status == "ok"                           # health is open
    alice = sync_client(app, protocol, api_key="secret", user_id="alice")
    bob = sync_client(app, protocol, api_key="secret", user_id="bob")
    d = alice.register_document(book, poll_interval=0.02).doc_id
    assert alice.get_document(d).doc_id == d and alice.get_learner(d).concepts
    for call in (lambda: bob.get_document(d), lambda: bob.get_structure(d), lambda: bob.get_knowledge(d), lambda: bob.get_learner(d),
                 lambda: bob.ask(d, "x"), lambda: bob.practice_quiz(d)):
        with pytest.raises(sdk.NotFoundError):
            call()
    assert bob.list_documents() == []


def test_a_protocol_the_server_does_not_serve_is_a_clear_error(tmp_path):
    rest_only = make_app(tmp_path / "a", enable_rpc=False)
    with pytest.raises(sdk.DocExplainerError) as e:
        sync_client(rest_only, "rpc").health()
    assert e.value.status == 404
    rpc_only = make_app(tmp_path / "b", enable_rest=False)
    assert sync_client(rpc_only, "rpc").health().protocols == ["rpc"]
    with pytest.raises(sdk.DocExplainerError):
        sync_client(rpc_only, "rest").health()
