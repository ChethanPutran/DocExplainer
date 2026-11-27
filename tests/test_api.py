"""The HTTP API (REST and JSON-RPC) over the REAL system: real stores, vector index and orchestration. Only the model is scripted."""
import base64
import threading
import time

import pytest
from fastapi.testclient import TestClient

from doc_explainer.api import ApiSettings, create_app
from doc_explainer.api.service import DocumentApiService
from doc_explainer.orchestrator import System, SystemConfig

from fakes import BOOK, Brain, HashingEmbedder

_services = []


@pytest.fixture(autouse=True)
def _close_services():
    yield
    while _services:
        _services.pop().close()


@pytest.fixture
def brain():
    return Brain()


def make_client(tmp_path, brain=None, **kw):
    settings = ApiSettings(data_dir=tmp_path / "api", **kw)
    system = System(SystemConfig(data_dir=tmp_path / "system"), llm=brain or Brain(), embedder=HashingEmbedder())
    service = DocumentApiService(system, settings)
    _services.append(service)
    return TestClient(create_app(service=service, settings=settings))


@pytest.fixture
def client(tmp_path, brain):
    return make_client(tmp_path, brain)


def rpc(client, method, params=None, id_=1, **kw):
    body = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        body["params"] = params
    return client.post("/rpc", json=body, **kw)


def upload(client, name="book.md", data=BOOK.encode(), **kw):
    """POST the file; returns the raw response (a 202 job, or a synchronous rejection)."""
    return client.post("/v1/documents", files={"file": (name, data)}, **kw)


def wait_job(client, job_id, timeout=15, **kw):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/v1/jobs/{job_id}", **kw).json()
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def register(client, name="book.md", data=BOOK.encode(), **kw):
    r = upload(client, name, data, **kw)
    assert r.status_code == 202, r.text
    return wait_job(client, r.json()["job_id"], **{k: v for k, v in kw.items() if k == "headers"})


def doc_id(client, **kw):
    return register(client, **kw)["document"]["doc_id"]


def ordinal(client, d, title, **kw):
    sections = client.get(f"/v1/documents/{d}/structure", **kw).json()["sections"]
    return next(s["ordinal"] for s in sections if s["title"] == title)


# ----------------------------------------------------------------------------- the full flow, over REST
def test_rest_full_flow(client):
    r = upload(client)
    assert r.status_code == 202 and r.headers["location"] == f"/v1/jobs/{r.json()['job_id']}"
    job = wait_job(client, r.json()["job_id"])
    assert job["state"] == "succeeded" and job["error"] is None and job["progress"] == 1.0 and job["stage"] == "done"
    doc = job["document"]
    assert (doc["title"], doc["filename"], doc["knowledge"], doc["sections"], doc["concepts"]) == ("Photosynthesis", "book.md", "ready", 4, 8)
    d = doc["doc_id"]
    assert set(doc) == {"doc_id", "title", "filename", "size_bytes", "uploaded_at", "knowledge", "sections", "concepts"}   # no paths/hashes
    assert client.get("/v1/documents").json()["documents"][0]["doc_id"] == d and client.get(f"/v1/documents/{d}").json()["title"] == "Photosynthesis"

    s = client.get(f"/v1/documents/{d}/structure").json()
    assert [(x["title"], x["depth"]) for x in s["sections"]] == [("Photosynthesis", 1), ("Light reactions", 2), ("Calvin cycle", 2), ("Cellular respiration", 2)]
    calvin = ordinal(client, d, "Calvin cycle")
    assert client.get(f"/v1/documents/{d}/section", params={"page": 1, "position": 0}).json()["section_id"] in {x["ordinal"] for x in s["sections"]}

    ex = client.post(f"/v1/documents/{d}/explain", json={"selected_text": "The Calvin cycle uses ATP.", "section_id": calvin})
    assert ex.status_code == 200, ex.text
    e = ex.json()["explanation"]
    assert ex.json()["text"].startswith("LEVEL=") and e["context_used"]["section"] == "Photosynthesis > Calvin cycle"
    assert e["style"]["level"] == "intermediate" and e["follow_up_questions"] == ["Why?"] and e["context_used"]["concept_ids"]
    assert client.post(f"/v1/documents/{d}/summarize", json={"selected_text": "x", "section_id": calvin}).json()["text"].startswith("LEVEL=")
    assert client.post(f"/v1/documents/{d}/ask", json={"question": "What is ATP?", "section_id": calvin}).status_code == 200
    forced = client.post(f"/v1/documents/{d}/explain", json={"selected_text": "ATP", "section_id": calvin, "style": {"level": "expert", "depth": "fixed"}})
    assert forced.json()["text"].startswith("LEVEL=expert")                                 # an explicit level wins
    ctx = client.get(f"/v1/documents/{d}/context", params={"section_id": calvin}).json()
    assert ctx["document_context"]["introduces"] and "ATP" in ctx["user_knowledge"] and ctx["session_data"]["position"] > calvin

    k = client.get(f"/v1/documents/{d}/knowledge").json()
    assert len(k["concepts"]) == 8 and k["as_of"] is None and any(r["type"] == "prerequisite_of" for r in k["relations"])
    early = client.get(f"/v1/documents/{d}/knowledge", params={"upto": ordinal(client, d, "Light reactions") + 5}).json()
    assert 0 < len(early["concepts"]) < 8 and early["as_of"] is not None                    # the graph as of a position
    sk = client.get(f"/v1/documents/{d}/sections/{calvin}/knowledge").json()
    assert {c["name"] for c in sk["introduces"]} == {"Calvin cycle", "carbon dioxide", "sugar"}
    assert [x["title"] for x in sk["depends_on"]] == ["Light reactions"]
    assert [p["total_concepts"] for p in client.get(f"/v1/documents/{d}/growth").json()["points"]] == [2, 4, 7, 8]
    pre = client.get(f"/v1/documents/{d}/concepts/cellular respiration/prerequisites").json()
    assert [(x["name"], x["distance"]) for x in pre["data"]] == [("sugar", 1), ("carbon dioxide", 2)]
    assert [x["name"] for x in client.get(f"/v1/documents/{d}/concepts/cellular respiration/learning-path").json()["data"]] == ["carbon dioxide", "sugar", "cellular respiration"]
    assert client.get(f"/v1/documents/{d}/concepts/photosynthesis/recommendations").json()["data"]
    assert client.delete(f"/v1/documents/{d}").status_code == 204
    assert client.get(f"/v1/documents/{d}").status_code == 404


def test_the_learner_flow_diagnostic_quiz_feedback_progress(client):
    d = doc_id(client)
    calvin = ordinal(client, d, "Calvin cycle")
    text = {"selected_text": "The Calvin cycle uses ATP from the light reactions.", "section_id": calvin}
    diag = client.post(f"/v1/documents/{d}/diagnostic", json={"level": "beginner", "known_topics": ["photosynthesis"], "questions": 4})
    assert diag.status_code == 200 and diag.json()["level"] == "beginner" and len(diag.json()["questions"]) == 4
    assert all(set(q) == {"id", "concept_id", "concept", "question", "options", "difficulty", "kind"} for q in diag.json()["questions"])   # no answer leaked
    assert client.post(f"/v1/documents/{d}/explain", json=text).json()["text"].startswith("LEVEL=beginner")
    for q in diag.json()["questions"]:                                                       # the brain always makes option 0 correct before shuffling; find it by trying
        for choice in range(4):
            res = client.post(f"/v1/documents/{d}/quiz/{q['id']}/answer", json={"choice": choice}).json()
            if choice == 0 or res["already_answered"]:
                pass
            if not res["already_answered"] or res["correct"]:
                break
    # (the question is answered once; re-answering never changes it)
    again = client.post(f"/v1/documents/{d}/quiz/{diag.json()['questions'][0]['id']}/answer", json={"choice": 0}).json()
    assert again["already_answered"] is True
    fin = client.post(f"/v1/documents/{d}/diagnostic/finish").json()
    assert fin["questions"] == [] and 0 < fin["prior"] < 1
    L = client.get(f"/v1/documents/{d}/learner").json()
    assert L["level"] == fin["level"] and L["concepts"] and all(0 <= c["p_known"] <= 1 for c in L["concepts"])
    quiz = client.post(f"/v1/documents/{d}/quiz", json={"section_id": calvin, "count": 2})
    assert quiz.status_code == 200 and 1 <= len(quiz.json()["questions"]) <= 2 and quiz.json()["questions"][0]["kind"] == "practice"
    cid = L["concepts"][0]["concept_id"]
    assert client.post(f"/v1/documents/{d}/feedback", json={"verdict": "too_hard", "concept_ids": [cid]}).json() == {"ok": True}
    assert client.post(f"/v1/documents/{d}/progress", json={"section_id": calvin}).json() == {"ok": True}
    assert client.get(f"/v1/documents/{d}/learner").json()["position"] is not None


def test_answering_a_quiz_question_is_graded_server_side_and_counts_once(client):
    d = doc_id(client)
    q = client.post(f"/v1/documents/{d}/quiz", json={"count": 1}).json()["questions"][0]
    results = {c: client.post(f"/v1/documents/{d}/quiz/{q['id']}/answer", json={"choice": c}).json() for c in range(4)}
    first = results[0]
    assert first["already_answered"] is False and {r["correct_index"] for r in results.values()} == {first["correct_index"]}
    assert all(r["already_answered"] for c, r in results.items() if c != 0)                 # only the first answer counts
    assert client.post(f"/v1/documents/{d}/quiz/{q['id']}/answer", json={"choice": 9}).status_code == 422
    r = client.post(f"/v1/documents/{d}/quiz/nope/answer", json={"choice": 0})
    assert r.status_code == 404 and r.json()["error"]["code"] == "question_not_found"


def test_health_is_open_even_with_api_key(tmp_path):
    c = make_client(tmp_path, api_key="secret")
    h = c.get("/v1/health").json()
    assert h["status"] == "ok" and h["api_version"] == "v1" and h["protocols"] == ["rest", "rpc"] and h["jobs_pending"] == 0
    assert {".pdf", ".md", ".txt", ".docx", ".html", ".pptx"} <= set(h["supported_types"])


def test_rest_error_envelope(client):
    assert client.get("/v1/documents/nope").json() == {"error": {"code": "document_not_found", "message": "Document 'nope' not found."}}
    assert upload(client, name="evil.exe").status_code == 415
    r = upload(client, name="notes.xyz")
    assert r.status_code == 415 and ".md" in r.json()["error"]["message"]
    assert upload(client, data=b"").status_code == 422
    d = doc_id(client)
    r = client.post(f"/v1/documents/{d}/explain", json={"selected_text": ""})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_params" and r.json()["error"]["details"][0]["field"] == "selected_text"
    r = client.get(f"/v1/documents/{d}/concepts/unobtainium/prerequisites")
    assert r.status_code == 404 and r.json()["error"]["code"] == "concept_not_found"
    assert client.get(f"/v1/documents/{d}/sections/0/knowledge").status_code in (200, 422)
    assert client.get(f"/v1/documents/{d}/sections/1/knowledge").status_code == 200            # a section ordinal
    assert client.get(f"/v1/documents/{d}/sections/3/knowledge").status_code == 422             # not a section's ordinal


def test_every_supported_format_can_be_uploaded(client):
    html = b"<html><head><title>Rocks</title></head><body><h1>Rocks</h1><p>Rocks are made of minerals.</p></body></html>"
    for name, data in (("a.txt", b"ROCKS\n\nRocks are made of minerals and more.\n"), ("a.html", html), ("a.md", b"# Rocks\n\nRocks are made of minerals.\n")):
        job = register(client, name, data + name.encode())
        assert job["state"] == "succeeded" and job["document"]["sections"] >= 1, (name, job)


def test_upload_size_limit(tmp_path, brain):
    c = make_client(tmp_path, brain, max_upload_mb=1)
    assert upload(c, data=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert upload(c, data=b"# T\n\nsmall text here.\n").status_code == 202


def test_filename_path_traversal_is_neutralised(client, tmp_path):
    job = register(client, name="../../etc/passwd.md", data=b"# T\n\nSome text about passwords.\n")
    assert job["document"]["filename"] == "passwd.md"
    stored = [p for p in (tmp_path / "api" / "uploads").rglob("passwd.md")]
    assert all((tmp_path / "api") in p.parents for p in stored)


# ----------------------------------------------------------------------------- jobs
def test_job_lifecycle_queued_running_succeeded_with_real_progress(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    books = [BOOK + f"\n\nExtra paragraph number {i} so that each upload is different.\n" for i in range(3)]
    first = upload(c, data=books[0].encode()).json()
    deadline = time.time() + 10
    while time.time() < deadline and not any("build a knowledge graph" in s for s, _ in brain.calls):
        time.sleep(0.02)                                                                        # job 1 is now blocked inside the knowledge stage
    running = c.get(f"/v1/jobs/{first['job_id']}").json()
    assert running["state"] == "running" and 0.35 <= running["progress"] < 1.0 and running["stage"].startswith("reading")
    second, third = (upload(c, data=b.encode()).json() for b in books[1:])
    assert c.get(f"/v1/jobs/{second['job_id']}").json()["queue_position"] == 1
    assert c.get(f"/v1/jobs/{third['job_id']}").json()["queue_position"] == 2
    assert c.get("/v1/health").json()["jobs_pending"] == 3
    assert [j["state"] for j in c.get("/v1/jobs").json()["jobs"]] == ["queued", "queued", "running"]
    brain.gate.set()
    done = [wait_job(c, j["job_id"]) for j in (first, second, third)]
    assert [j["state"] for j in done] == ["succeeded"] * 3 and all(j["queue_position"] is None for j in done)
    assert c.get("/v1/health").json()["jobs_pending"] == 0


def test_a_corrupt_file_fails_the_job_with_the_reason_and_leaves_nothing_behind(client, tmp_path):
    job = register(client, name="broken.pdf", data=b"this is not a pdf")
    assert job["state"] == "failed" and job["document"] is None
    assert job["error"]["code"] == "processing_failed" and "broken.pdf" in job["error"]["message"]
    assert client.get("/v1/documents").json()["documents"] == []
    assert not any((tmp_path / "api" / "uploads").rglob("*.pdf"))


def test_a_scanned_pdf_with_no_text_says_what_to_do(client):
    import pymupdf
    doc = pymupdf.open(); doc.new_page(); data = doc.tobytes(); doc.close()
    job = register(client, name="scan.pdf", data=data)
    assert job["state"] == "failed" and "OCR" in job["error"]["message"]


def test_a_failing_model_still_gives_a_usable_document_and_says_so(tmp_path, brain):
    brain.fail_knowledge = True
    c = make_client(tmp_path, brain)
    job = register(c)
    assert job["state"] == "succeeded" and job["document"]["knowledge"] == "failed" and job["document"]["concepts"] == 0
    d = job["document"]["doc_id"]
    brain.fail_knowledge = False
    assert c.post(f"/v1/documents/{d}/ask", json={"question": "What is photosynthesis?"}).status_code == 200   # still answers questions


def test_uploading_a_document_whose_knowledge_graph_failed_retries_it_on_the_same_document(tmp_path, brain):
    brain.fail_knowledge = True
    c = make_client(tmp_path, brain)
    first = register(c)
    d = first["document"]["doc_id"]
    assert first["document"]["knowledge"] == "failed"
    brain.fail_knowledge = False                                                         # the model works now
    again = register(c)
    assert not again["reused"] and again["document"]["doc_id"] == d                      # same document, not a duplicate
    assert again["document"]["knowledge"] == "ready" and again["document"]["concepts"] == 8
    assert len(c.get("/v1/documents").json()["documents"]) == 1
    assert upload(c).json()["reused"] is True                                            # healthy now: instant again


def test_a_document_registered_without_a_graph_gets_one_later_on_the_same_document(client):
    job = wait_job(client, client.post("/v1/documents", files={"file": ("book.md", BOOK.encode())}, data={"build_graph": "false"}).json()["job_id"])
    d = job["document"]["doc_id"]
    assert job["document"]["knowledge"] == "none" and job["document"]["concepts"] == 0
    assert client.post("/v1/documents", files={"file": ("book.md", BOOK.encode())}, data={"build_graph": "false"}).json()["reused"] is True        # asking for no graph again: instant
    later = wait_job(client, upload(client).json()["job_id"])                               # now with a graph
    assert not later["reused"] and later["document"]["doc_id"] == d                         # same document, not a copy
    assert later["document"]["knowledge"] == "ready" and later["document"]["concepts"] == 8
    assert len(client.get("/v1/documents").json()["documents"]) == 1


def test_a_retry_that_fails_again_keeps_the_existing_document(tmp_path, brain):
    brain.fail_knowledge = True
    c = make_client(tmp_path, brain)
    d = register(c)["document"]["doc_id"]
    again = register(c)                                                                  # still failing
    assert again["state"] == "succeeded" and again["document"]["doc_id"] == d and again["document"]["knowledge"] == "failed"
    assert c.get(f"/v1/documents/{d}").status_code == 200
    assert c.get(f"/v1/documents/{d}/structure").status_code == 200                      # nothing was deleted
    assert c.post(f"/v1/documents/{d}/ask", json={"question": "What is photosynthesis?"}).status_code == 200


def test_a_retry_that_cannot_read_the_file_does_not_delete_the_existing_document(tmp_path, brain, monkeypatch):
    brain.fail_knowledge = True
    c = make_client(tmp_path, brain)
    d = register(c)["document"]["doc_id"]
    service = c.app.state.service
    monkeypatch.setattr(service._sys, "ingest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    job = register(c)
    assert job["state"] == "failed"
    assert c.get(f"/v1/documents/{d}").status_code == 200 and c.get(f"/v1/documents/{d}/structure").status_code == 200


def test_questions_keep_working_while_another_document_is_being_processed(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)                                                                               # document A is ready
    brain.gate = threading.Event()                                                              # B's knowledge stage will block
    other = upload(c, data=b"# Rocks\n\nRocks are made of minerals.\n\n## More\n\nMore about rocks and geology.\n").json()
    deadline = time.time() + 10
    while time.time() < deadline and not any("geology" in u for _, u in brain.calls):
        time.sleep(0.02)
    assert c.get(f"/v1/jobs/{other['job_id']}").json()["state"] == "running"
    t0 = time.time()
    r = c.post(f"/v1/documents/{d}/ask", json={"question": "What is the Calvin cycle?", "section_id": ordinal(c, d, "Calvin cycle")})
    assert r.status_code == 200 and time.time() - t0 < 3                                         # no waiting, no "busy"
    assert c.get(f"/v1/documents/{d}/learner").status_code == 200 and c.get(f"/v1/documents/{d}/knowledge").status_code == 200
    brain.gate.set()
    assert wait_job(c, other["job_id"])["state"] == "succeeded"


def test_queue_limits(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain, max_jobs_per_user=2, max_queued_jobs=3)
    h = lambda u: {"X-User-Id": u}
    body = lambda tag: (BOOK + f"\n\n{tag}\n").encode()
    assert upload(c, data=body("a"), headers=h("amy")).status_code == 202
    assert upload(c, data=body("b"), headers=h("amy")).status_code == 202
    r = upload(c, data=body("c"), headers=h("amy"))
    assert r.status_code == 429 and r.json()["error"]["code"] == "queue_full"
    assert upload(c, data=body("d"), headers=h("bob")).status_code == 202
    assert upload(c, data=body("e"), headers=h("cat")).status_code == 429
    brain.gate.set()


def test_identical_upload_is_not_processed_twice(client, brain):
    first = register(client)
    calls = len(brain.calls)
    again = upload(client).json()
    assert again["state"] == "succeeded" and again["reused"] is True and again["document"]["doc_id"] == first["document"]["doc_id"]
    assert len(brain.calls) == calls and len(client.get("/v1/documents").json()["documents"]) == 1
    assert not register(client, data=(BOOK + "\n\nDifferent now.\n").encode())["reused"]


def test_identical_upload_while_processing_returns_the_running_job(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    a, b = upload(c).json(), upload(c).json()
    assert a["job_id"] == b["job_id"]
    brain.gate.set()
    wait_job(c, a["job_id"])


def test_jobs_interrupted_by_a_restart_are_reported_failed(tmp_path, brain):
    brain.gate = threading.Event()
    c = make_client(tmp_path, brain)
    job = upload(c).json()
    deadline = time.time() + 10
    while time.time() < deadline and not brain.calls:
        time.sleep(0.02)
    settings = ApiSettings(data_dir=tmp_path / "api")                                           # a new server over the same data while it ran
    from doc_explainer.api.jobs import JobStore
    restarted = JobStore(settings.data_dir / "jobs.json")
    j = restarted.get(job["job_id"], "default")
    assert j["state"] == "failed" and "restarted" in j["error"]["message"]
    brain.gate.set()


# ----------------------------------------------------------------------------- privacy and persistence
def test_everything_is_private_per_user(client):
    h = {"X-User-Id": "alice"}
    job = register(client, headers=h)
    d = job["document"]["doc_id"]
    bob = {"X-User-Id": "bob"}
    probes = [("get", f"/v1/documents/{d}"), ("get", f"/v1/documents/{d}/structure"), ("get", f"/v1/documents/{d}/knowledge"),
              ("get", f"/v1/documents/{d}/growth"), ("get", f"/v1/documents/{d}/learner"),
              ("get", f"/v1/documents/{d}/concepts/sugar/prerequisites"), ("get", f"/v1/jobs/{job['job_id']}"),
              ("post", f"/v1/documents/{d}/quiz", {"count": 1}), ("post", f"/v1/documents/{d}/ask", {"question": "x"}),
              ("post", f"/v1/documents/{d}/diagnostic", {"level": "beginner"}), ("delete", f"/v1/documents/{d}")]
    for method, url, *body in probes:
        r = getattr(client, method)(url, headers=bob, **({"json": body[0]} if body else {}))
        assert r.status_code == 404, (method, url, r.status_code)
    assert client.get("/v1/documents", headers=bob).json()["documents"] == [] and client.get("/v1/jobs", headers=bob).json()["jobs"] == []
    assert client.get(f"/v1/documents/{d}", headers=h).status_code == 200
    assert upload(client, headers=bob).json()["reused"] is False                                 # same bytes, another user: their own copy
    assert client.get("/v1/documents", headers={"X-User-Id": "../x"}).status_code == 422


def test_a_learners_progress_is_theirs_alone(client):
    d = doc_id(client, headers={"X-User-Id": "alice"})
    client.post(f"/v1/documents/{d}/ask", json={"question": "What is ATP?"}, headers={"X-User-Id": "alice"})
    assert client.get(f"/v1/documents/{d}/learner", headers={"X-User-Id": "alice"}).json()["concepts"]
    # bob owns no copy of that document, but even a document he uploaded identically has its own learner state
    d2 = doc_id(client, headers={"X-User-Id": "bob"})
    assert d2 != d
    assert {c["evidence"] for c in client.get(f"/v1/documents/{d2}/learner", headers={"X-User-Id": "bob"}).json()["concepts"]} == {0}


def test_everything_survives_a_restart(tmp_path):
    first = make_client(tmp_path, Brain())
    d = doc_id(first)
    first.post(f"/v1/documents/{d}/ask", json={"question": "What is ATP?"})
    before = first.get(f"/v1/documents/{d}/learner").json()
    after_restart = make_client(tmp_path, Brain())                                               # same data, new processes
    assert after_restart.get(f"/v1/documents/{d}").json()["title"] == "Photosynthesis"
    assert len(after_restart.get(f"/v1/documents/{d}/knowledge").json()["concepts"]) == 8
    assert after_restart.get(f"/v1/documents/{d}/learner").json() == before                      # the learner model persisted
    assert after_restart.post(f"/v1/documents/{d}/ask", json={"question": "Why does ATP matter?"}).status_code == 200


# ----------------------------------------------------------------------------- auth
def test_api_key_required_on_both_protocols(tmp_path):
    c = make_client(tmp_path, api_key="secret")
    assert c.get("/v1/documents").status_code == 401 and c.get("/v1/jobs").status_code == 401
    assert c.get("/v1/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert rpc(c, "document.list", {}).status_code == 401
    ok = {"Authorization": "Bearer secret"}
    assert c.get("/v1/documents", headers=ok).status_code == 200
    assert rpc(c, "document.list", {}, headers=ok).json()["result"] == {"documents": []}


# ----------------------------------------------------------------------------- RPC
def rpc_wait(client, job_id, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        job = rpc(client, "job.get", {"job_id": job_id}).json()["result"]
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_rpc_full_flow(client):
    b64 = base64.b64encode(BOOK.encode()).decode()
    r = rpc(client, "document.register", {"filename": "book.md", "content_b64": b64}).json()
    assert r["id"] == 1 and r["result"]["state"] in ("queued", "running", "succeeded")
    job = rpc_wait(client, r["result"]["job_id"])
    d = job["document"]["doc_id"]
    assert job["state"] == "succeeded" and job["document"]["knowledge"] == "ready"
    assert [j["job_id"] for j in rpc(client, "job.list", {}).json()["result"]["jobs"]] == [job["job_id"]]
    assert rpc(client, "document.get", {"doc_id": d}).json()["result"]["title"] == "Photosynthesis"
    sections = rpc(client, "document.structure", {"doc_id": d}).json()["result"]["sections"]
    calvin = next(s["ordinal"] for s in sections if s["title"] == "Calvin cycle")
    assert rpc(client, "document.section_at", {"doc_id": d, "page": 1, "position": 0}).json()["result"]["section_id"] in {s["ordinal"] for s in sections}
    ex = rpc(client, "document.explain", {"doc_id": d, "selected_text": "ATP", "section_id": calvin, "style": {"level": "expert", "depth": "adaptive"}}).json()["result"]
    assert ex["text"].startswith("LEVEL=expert")
    assert rpc(client, "document.summarize", {"doc_id": d, "selected_text": "x", "section_id": calvin}).json()["result"]["text"]
    assert rpc(client, "document.ask", {"doc_id": d, "question": "What is ATP?", "section_id": calvin}).json()["result"]["text"]
    assert rpc(client, "document.context", {"doc_id": d, "section_id": calvin}).json()["result"]["document_context"]["introduces"]
    assert len(rpc(client, "document.knowledge", {"doc_id": d}).json()["result"]["concepts"]) == 8
    assert rpc(client, "document.section_knowledge", {"doc_id": d, "section_id": calvin}).json()["result"]["depends_on"]
    assert len(rpc(client, "document.growth", {"doc_id": d}).json()["result"]["points"]) == 4
    assert rpc(client, "concept.prerequisites", {"doc_id": d, "name": "cellular respiration"}).json()["result"]["data"][0]["name"] == "sugar"
    assert rpc(client, "concept.learning_path", {"doc_id": d, "name": "cellular respiration"}).json()["result"]["data"][-1]["name"] == "cellular respiration"
    assert rpc(client, "concept.recommendations", {"doc_id": d, "name": "photosynthesis"}).json()["result"]["data"]
    diag = rpc(client, "learner.diagnostic_start", {"doc_id": d, "level": "beginner", "questions": 3}).json()["result"]
    assert len(diag["questions"]) == 3
    ans = rpc(client, "quiz.answer", {"doc_id": d, "question_id": diag["questions"][0]["id"], "choice": 0}).json()["result"]
    assert set(ans) == {"correct", "correct_index", "explanation", "already_answered"}
    assert rpc(client, "learner.diagnostic_finish", {"doc_id": d}).json()["result"]["questions"] == []
    assert rpc(client, "quiz.new", {"doc_id": d, "count": 2}).json()["result"]["questions"]
    assert rpc(client, "learner.get", {"doc_id": d}).json()["result"]["concepts"]
    cid = rpc(client, "learner.get", {"doc_id": d}).json()["result"]["concepts"][0]["concept_id"]
    assert rpc(client, "learner.feedback", {"doc_id": d, "verdict": "too_easy", "concept_ids": [cid]}).json()["result"] == {"ok": True}
    assert rpc(client, "learner.progress", {"doc_id": d, "section_id": calvin}).json()["result"] == {"ok": True}
    assert rpc(client, "document.delete", {"doc_id": d}).json()["result"] == {"ok": True}
    assert rpc(client, "system.health").json()["result"]["protocols"] == ["rest", "rpc"]


def test_rest_and_rpc_return_identical_results(client):
    d = doc_id(client)
    calvin = ordinal(client, d, "Calvin cycle")
    pairs = [
        (client.get(f"/v1/documents/{d}/structure").json(), rpc(client, "document.structure", {"doc_id": d}).json()["result"]),
        (client.get(f"/v1/documents/{d}/knowledge").json(), rpc(client, "document.knowledge", {"doc_id": d}).json()["result"]),
        (client.get(f"/v1/documents/{d}/growth").json(), rpc(client, "document.growth", {"doc_id": d}).json()["result"]),
        (client.get(f"/v1/documents/{d}/sections/{calvin}/knowledge").json(), rpc(client, "document.section_knowledge", {"doc_id": d, "section_id": calvin}).json()["result"]),
        (client.get(f"/v1/documents/{d}/concepts/cellular respiration/prerequisites").json(), rpc(client, "concept.prerequisites", {"doc_id": d, "name": "cellular respiration"}).json()["result"]),
    ]
    for rest, via_rpc in pairs:
        assert rest == via_rpc
    body = {"selected_text": "The Calvin cycle uses ATP.", "section_id": calvin}
    a = client.post(f"/v1/documents/{d}/explain", json=body).json()
    b = rpc(client, "document.explain", {"doc_id": d, **body}).json()["result"]
    for x in (a, b):
        x["explanation"]["metadata"]["processing_time_ms"] = 0
    assert a == b


def test_rpc_errors(client):
    err = lambda r: r.json()["error"]
    d = doc_id(client)
    assert err(rpc(client, "nope"))["code"] == -32601
    e = err(rpc(client, "document.get", {"doc_id": "zzz"}))
    assert (e["code"], e["data"]["code"]) == (-32004, "document_not_found")
    assert (err(rpc(client, "job.get", {"job_id": "zzz"}))["code"]) == -32008
    assert err(rpc(client, "quiz.answer", {"doc_id": d, "question_id": "zzz", "choice": 0}))["code"] == -32010
    assert err(rpc(client, "concept.prerequisites", {"doc_id": d, "name": "unobtainium"}))["code"] == -32011
    e = err(rpc(client, "document.explain", {"doc_id": d, "selected_text": ""}))
    assert e["code"] == -32602 and e["data"]["details"][0]["field"] == "selected_text"
    assert err(rpc(client, "document.get", {"doc_id": d, "extra": 1}))["code"] == -32602
    assert err(rpc(client, "document.get", ["positional"]))["code"] == -32602
    assert err(client.post("/rpc", content=b"{not json"))["code"] == -32700
    assert err(client.post("/rpc", json={"method": "x"}))["code"] == -32600 and err(client.post("/rpc", json=[]))["code"] == -32600
    assert err(rpc(client, "document.register", {"filename": "a.md", "content_b64": "!!notb64!!"}))["data"]["code"] == "invalid_params"
    e = err(rpc(client, "document.register", {"filename": "a.exe", "content_b64": base64.b64encode(b"x").decode()}))
    assert (e["code"], e["data"]["code"]) == (-32006, "unsupported_file_type")
    job = rpc_wait(client, rpc(client, "document.register", {"filename": "broken.pdf", "content_b64": base64.b64encode(b"nope").decode()}).json()["result"]["job_id"])
    assert job["state"] == "failed" and job["error"]["code"] == "processing_failed"


def test_rpc_notifications_and_batches(client):
    assert client.post("/rpc", json={"jsonrpc": "2.0", "method": "system.health"}).status_code == 204
    batch = [{"jsonrpc": "2.0", "method": "system.health", "id": "a"}, {"jsonrpc": "2.0", "method": "system.health"},
             {"jsonrpc": "2.0", "method": "nope", "id": "b"}]
    out = client.post("/rpc", json=batch).json()
    assert [o["id"] for o in out] == ["a", "b"] and "result" in out[0] and out[1]["error"]["code"] == -32601
    assert "error" in client.post("/rpc", json=[{"jsonrpc": "2.0", "method": "system.health", "id": i} for i in range(21)]).json()


def test_rpc_discover_lists_every_method(client):
    from doc_explainer.api.rpc import METHODS
    found = rpc(client, "rpc.discover").json()["result"]
    assert set(found) == set(METHODS) and "doc_id" in found["document.explain"]["params"]["properties"]
    assert {"job.get", "quiz.new", "quiz.answer", "learner.get", "document.knowledge", "document.structure"} <= set(found)


# ----------------------------------------------------------------------------- toggles
def test_protocols_can_be_disabled_independently(tmp_path, brain):
    rest_only = make_client(tmp_path / "a", brain, enable_rpc=False)
    assert rest_only.post("/rpc", json={}).status_code == 404 and rest_only.get("/v1/health").json()["protocols"] == ["rest"]
    rpc_only = make_client(tmp_path / "b", Brain(), enable_rest=False)
    assert rpc_only.get("/v1/health").status_code == 404 and rpc(rpc_only, "system.health").json()["result"]["protocols"] == ["rpc"]


def test_not_ready_before_service_exists(tmp_path):
    app = create_app(service=None, settings=ApiSettings(data_dir=tmp_path))
    r = TestClient(app).get("/v1/health")
    assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"


# ---------------------------------------------------------------- library maintenance, review, export
def test_retention_decays_with_time_and_slower_with_more_evidence():
    from doc_explainer.api.learning import retention
    assert retention(0.9, 1, 0) == pytest.approx(0.9)
    assert retention(0.9, 1, 2) == pytest.approx(0.45)                       # one half-life
    assert retention(0.9, 4, 2) > retention(0.9, 1, 2)                       # well-practised concepts fade slower


def test_review_lists_practised_concepts_and_marks_the_forgotten_ones_due(client):
    from doc_explainer.api import learning
    d = doc_id(client)
    assert client.get(f"/v1/documents/{d}/review").json()["items"] == []        # nothing practised yet: nothing to review
    diag = client.post(f"/v1/documents/{d}/diagnostic", json={"level": "intermediate", "known_topics": [], "questions": 3}).json()
    for q in diag["questions"]:
        client.post(f"/v1/documents/{d}/quiz/{q['id']}/answer", json={"choice": 0})
    fresh = client.get(f"/v1/documents/{d}/review").json()["items"]
    assert fresh and all(i["due"] == (i["retention"] < 0.6) for i in fresh)       # due exactly when retention is low
    assert [i["retention"] for i in fresh] == sorted(i["retention"] for i in fresh)       # weakest first
    later = _services[-1].learning.review("default", d, 50, now_ts=time.time() + 60 * 86400)       # two months on
    assert later.items and all(i.due for i in later.items)
    assert client.get(f"/v1/documents/{d}/history").json()["points"]             # the answers left a trail


def test_deleting_the_knowledge_graph_keeps_the_document_usable(client):
    d = doc_id(client)
    assert client.get(f"/v1/documents/{d}").json()["knowledge"] == "ready"
    assert client.delete(f"/v1/documents/{d}/knowledge").status_code == 204
    doc = client.get(f"/v1/documents/{d}").json()
    assert doc["knowledge"] == "none" and doc["concepts"] == 0
    assert client.post(f"/v1/documents/{d}/ask", json={"question": "What is photosynthesis?"}).status_code == 200
    assert wait_job(client, upload(client).json()["job_id"])["document"]["doc_id"] == d       # and it can be rebuilt in place
    assert client.get(f"/v1/documents/{d}").json()["knowledge"] == "ready"


def test_deleting_a_document_removes_its_stored_upload(tmp_path, brain):
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    uploads = tmp_path / "api" / "uploads"
    assert any(uploads.rglob("book.md"))
    assert c.delete(f"/v1/documents/{d}").status_code == 204
    assert not any(uploads.rglob("book.md"))


def test_a_conversation_exports_as_markdown(client):
    d = doc_id(client)
    sid = client.post(f"/v1/documents/{d}/sessions", json={"title": "My chat"}).json()["id"]
    client.post(f"/v1/documents/{d}/sessions/{sid}/notes", json={"text": "remember this", "kind": "note"})
    out = client.get(f"/v1/documents/{d}/sessions/{sid}/export").json()
    assert out["filename"] == "My chat.md" and out["markdown"].startswith("# My chat") and "remember this" in out["markdown"]
