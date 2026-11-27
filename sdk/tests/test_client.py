"""SDK unit tests: no backend. A mock transport records exactly what the SDK puts on the wire."""
import base64
import json

import httpx
import pytest

import doc_explainer_sdk as sdk

DOC = {"doc_id": "d1", "title": "T", "filename": "f.pdf", "size_bytes": 3, "uploaded_at": "now"}


def job(state="queued", **kw):
    base = {"job_id": "j1", "state": state, "filename": "f.pdf", "created_at": "now"}
    if state == "succeeded":
        base["document"] = DOC
    return {**base, **kw}


def make(handler, **kw):
    kw.setdefault("protocol", "rest")
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return sdk.DocExplainerClient("http://srv:8000/", http_client=http, **kw)


def test_rest_request_shape():
    seen = []

    def handler(req):
        seen.append(req)
        if req.url.path.endswith("/explain"):
            return httpx.Response(200, json={"text": "t", "explanation": {"explanation": "e"}})
        return httpx.Response(200, json={"section_id": 7})

    c = make(handler, api_key="k3y", user_id="alice")
    assert c.get_section_id("d/1", 2, 3) == 7
    r = seen[0]
    assert r.method == "GET"
    assert r.url.raw_path.decode().startswith("/v1/documents/d%2F1/section")  # id is quoted, not a path
    assert dict(r.url.params) == {"page": "2", "position": "3"}
    assert r.headers["authorization"] == "Bearer k3y" and r.headers["x-user-id"] == "alice"
    assert r.headers["user-agent"].startswith("doc-explainer-sdk/")

    c.explain("d1", "txt", section_id=1, style=sdk.ExplanationStyle(level="advanced", depth="fixed"))
    r = seen[1]
    assert (r.method, r.url.path) == ("POST", "/v1/documents/d1/explain")
    assert json.loads(r.content) == {"selected_text": "txt", "section_id": 1,
                                     "style": {"level": "advanced", "depth": "fixed", "description": ""}}


def test_rest_upload_is_multipart(tmp_path):
    seen = []
    f = tmp_path / "f.pdf"
    f.write_bytes(b"PDFDATA")
    c = make(lambda req: (seen.append(req), httpx.Response(202, json=job()))[1])
    assert c.start_registration(f, build_graph=False).job_id == "j1"   # returns without polling
    body = seen[0].content
    assert seen[0].headers["content-type"].startswith("multipart/form-data")
    assert b'filename="f.pdf"' in body and b"PDFDATA" in body and b"false" in body


def test_rpc_request_shape(tmp_path):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((req, body))
        result = job() if body["method"] == "document.register" else {"section_id": 1}
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": result, "id": body["id"]})

    c = make(handler, protocol="rpc", api_key="k")
    c.start_registration(b"abc", filename="f.pdf")
    req, body = seen[0]
    assert (req.method, req.url.path) == ("POST", "/rpc")
    assert req.headers["authorization"] == "Bearer k"
    assert body["jsonrpc"] == "2.0" and body["method"] == "document.register"
    assert body["params"] == {"filename": "f.pdf", "build_graph": True,
                              "content_b64": base64.b64encode(b"abc").decode()}
    c.get_section_id("d1", 1, 2)
    assert seen[1][1]["method"] == "document.section_at" and seen[1][1]["id"] == body["id"] + 1


def test_none_params_are_omitted():
    seen = []
    c = make(lambda req: (seen.append(req), httpx.Response(200, json={"text": "t", "explanation": {"explanation": "e"}}))[1])
    c.explain("d1", "x")  # no style
    assert "style" not in json.loads(seen[0].content)


def test_error_mapping_rest():
    def err(status, code, msg="m"):
        return lambda req: httpx.Response(status, json={"error": {"code": code, "message": msg}})

    for status, code, exc in [(404, "document_not_found", sdk.NotFoundError),
                              (401, "unauthorized", sdk.AuthenticationError),
                              (422, "invalid_params", sdk.InvalidRequestError),
                              (413, "file_too_large", sdk.InvalidRequestError),
                              (422, "processing_failed", sdk.ProcessingError),
                              (503, "not_ready", sdk.ServerUnavailableError),
                              (500, "internal_error", sdk.ServerError),
                              (418, "something_new", sdk.ServerError)]:
        with pytest.raises(exc) as e:
            make(err(status, code)).list_documents()
        assert e.value.code == code and e.value.status == status


def test_proxy_html_and_wrong_server_are_not_mistaken_for_api_errors():
    with pytest.raises(sdk.ServerUnavailableError):
        make(lambda req: httpx.Response(502, text="<html>Bad Gateway</html>")).list_documents()
    with pytest.raises(sdk.ServerError) as e:
        make(lambda req: httpx.Response(404, text="<html>nginx</html>")).list_documents()
    assert "is the URL right" in str(e.value)
    with pytest.raises(sdk.ProtocolError):
        make(lambda req: httpx.Response(200, text="not json")).list_documents()


def test_rpc_protocol_violations():
    wrong_id = lambda req: httpx.Response(200, json={"jsonrpc": "2.0", "result": {}, "id": 999})
    with pytest.raises(sdk.ProtocolError):
        make(wrong_id, protocol="rpc").list_documents()
    not_rpc = lambda req: httpx.Response(200, json={"hello": "world"})
    with pytest.raises(sdk.ProtocolError):
        make(not_rpc, protocol="rpc").list_documents()


def test_rpc_error_object_mapping():
    def rpc_err(code, data=None):
        return lambda req: httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error":
                                                      {"code": code, "message": "boom", "data": data}})
    with pytest.raises(sdk.NotFoundError):
        make(rpc_err(-32004, {"code": "document_not_found"}), protocol="rpc").list_documents()
    with pytest.raises(sdk.InvalidRequestError) as e:  # falls back to the numeric code
        make(rpc_err(-32602), protocol="rpc").list_documents()
    assert e.value.code == "invalid_params"
    with pytest.raises(sdk.AuthenticationError):  # rejected before dispatch, as plain HTTP
        make(lambda req: httpx.Response(401, json={"error": {"code": "unauthorized", "message": "no"}}),
             protocol="rpc").list_documents()


def test_connection_failures_become_server_unavailable():
    def refuse(req):
        raise httpx.ConnectError("refused")

    def slow(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(sdk.ServerUnavailableError) as e:
        make(refuse).health()
    assert "http://srv:8000" in str(e.value) and "Is it running" in str(e.value)
    with pytest.raises(sdk.ServerUnavailableError) as e:
        make(slow, timeout=5).health()
    assert "5s" in str(e.value)


def test_configuration(monkeypatch):
    with pytest.raises(ValueError):
        sdk.DocExplainerClient(protocol="grpc")
    monkeypatch.setenv("DOC_EXPLAINER_URL", "http://env-host:9/")
    monkeypatch.setenv("DOC_EXPLAINER_PROTOCOL", "RPC")
    monkeypatch.setenv("DOC_EXPLAINER_API_KEY", "envkey")
    c = sdk.DocExplainerClient()
    assert (c.base_url, c.protocol) == ("http://env-host:9", "rpc")
    assert c._headers["Authorization"] == "Bearer envkey"
    assert sdk.DocExplainerClient("http://x", api_key="").protocol == "rpc"
    assert "envkey" not in repr(c)
    assert "Authorization" not in sdk.DocExplainerClient("http://x", api_key="")._headers
    c.close()


def test_bytes_upload_needs_a_filename():
    with pytest.raises(ValueError):
        make(lambda req: httpx.Response(200, json=DOC)).register_document(b"abc")


async def test_async_client_error_and_unreachable():
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"code": "document_not_found", "message": "x"}})))
    async with sdk.AsyncDocExplainerClient("http://srv", http_client=http) as c:
        with pytest.raises(sdk.NotFoundError):
            await c.get_document("d")

    def refuse(req):
        raise httpx.ConnectError("refused")
    c = sdk.AsyncDocExplainerClient("http://srv", http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    with pytest.raises(sdk.ServerUnavailableError):
        await c.health()


# ----------------------------------------------------------------------------- background jobs
def rest_job_server(states, post=None):
    """Mock server: POST /v1/documents answers `post`; GET /v1/jobs/j1 walks through `states`."""
    polls = []

    def handler(req):
        if req.method == "POST":
            return httpx.Response(202, json=post or job("queued", queue_position=2))
        polls.append(req)
        s = states[min(len(polls) - 1, len(states) - 1)]
        return httpx.Response(200, json=s)

    return handler, polls


def test_register_document_waits_through_the_job_states():
    handler, polls = rest_job_server([job("queued", queue_position=1), job("queued", queue_position=1),
                                      job("running"), job("running"), job("succeeded")])
    updates = []
    doc = make(handler).register_document(b"abc", filename="f.pdf", poll_interval=0.001,
                                          on_update=lambda j: updates.append((j.state, j.queue_position)))
    assert doc.doc_id == "d1" and len(polls) == 5
    assert updates == [("queued", 2), ("queued", 1), ("running", None), ("succeeded", None)]  # changes only


def test_wait_for_job_returns_the_finished_job():
    handler, _ = rest_job_server([job("running"), job("succeeded")])
    j = make(handler).wait_for_job("j1", poll_interval=0.001)
    assert j.done and j.state == "succeeded" and j.document.title == "T"


@pytest.mark.parametrize("code,exc", [("processing_failed", sdk.ProcessingError),
                                      ("unsupported_file_type", sdk.InvalidRequestError),
                                      ("internal_error", sdk.ServerError)])
def test_failed_job_raises_the_mapped_error(code, exc):
    failed = job("failed", error={"code": code, "message": "nope"})
    handler, _ = rest_job_server([job("running"), failed])
    with pytest.raises(exc) as e:
        make(handler).register_document(b"abc", filename="f.pdf", poll_interval=0.001)
    assert e.value.code == code and e.value.message == "nope"


def test_reused_or_already_failed_job_needs_no_polling():
    handler, polls = rest_job_server([], post=job("succeeded", reused=True))
    assert make(handler).register_document(b"abc", filename="f.pdf").doc_id == "d1"
    assert polls == []
    handler, polls = rest_job_server([], post=job("failed", error={"code": "processing_failed", "message": "x"}))
    with pytest.raises(sdk.ProcessingError):
        make(handler).register_document(b"abc", filename="f.pdf")
    assert polls == []


def test_wait_timeout_leaves_the_job_running_and_resumable():
    handler, _ = rest_job_server([job("running")])
    with pytest.raises(sdk.JobTimeoutError) as e:
        make(handler).register_document(b"abc", filename="f.pdf", timeout=0.05, poll_interval=0.01)
    assert e.value.job_id == "j1" and e.value.job.state == "running" and "still running" in str(e.value)
    handler, _ = rest_job_server([job("succeeded")])           # ...and the caller can pick it up again
    assert make(handler).wait_for_job(e.value.job_id).document.doc_id == "d1"


def test_wait_survives_a_brief_outage_but_not_a_long_one():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("restarting")
        return httpx.Response(200, json=job("succeeded"))

    assert make(flaky).wait_for_job("j1", poll_interval=0.001).state == "succeeded"

    def dead(req):
        raise httpx.ConnectError("gone")
    with pytest.raises(sdk.ServerUnavailableError):
        make(dead).wait_for_job("j1", poll_interval=0.001, max_poll_errors=2)


def test_a_full_queue_is_a_retryable_unavailable_error():
    h = lambda req: httpx.Response(429, json={"error": {"code": "queue_full", "message": "later"}})
    with pytest.raises(sdk.ServerBusyError) as e:
        make(h).ask("d1", "q")
    assert isinstance(e.value, sdk.ServerUnavailableError) and e.value.code == "queue_full"


def test_unknown_job_is_not_found():
    h = lambda req: httpx.Response(404, json={"error": {"code": "job_not_found", "message": "x"}})
    with pytest.raises(sdk.NotFoundError):
        make(h).get_job("zzz")


def test_rpc_job_polling_uses_job_get():
    methods = []

    def handler(req):
        body = json.loads(req.content)
        methods.append(body["method"])
        result = job("queued") if body["method"] == "document.register" else job("succeeded")
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": result, "id": body["id"]})

    doc = make(handler, protocol="rpc").register_document(b"abc", filename="f.pdf", poll_interval=0.001)
    assert doc.doc_id == "d1" and methods == ["document.register", "job.get"]


def test_list_jobs():
    h = lambda req: httpx.Response(200, json={"jobs": [job("running"), job("succeeded", job_id="j0")]})
    assert [j.job_id for j in make(h).list_jobs()] == ["j1", "j0"]


async def test_async_register_waits_for_the_job():
    states = iter([job("running"), job("succeeded")])

    def handler(req):
        if req.method == "POST":
            return httpx.Response(202, json=job("queued"))
        return httpx.Response(200, json=next(states))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with sdk.AsyncDocExplainerClient("http://srv", http_client=http) as c:
        assert (await c.register_document(b"abc", filename="f.pdf", poll_interval=0.001)).doc_id == "d1"

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=job("running"))))
    async with sdk.AsyncDocExplainerClient("http://srv", http_client=http) as c:
        with pytest.raises(sdk.JobTimeoutError):
            await c.wait_for_job("j1", timeout=0.05, poll_interval=0.01)


# ----------------------------------------------------------------------------- the knowledge / learner operations
CONCEPT = {"id": "c1", "name": "ATP", "introduced_ordinal": 5, "section_id": "s1.1", "definition": "energy", "importance": 3, "aliases": []}
QUESTION = {"id": "q1", "concept_id": "c1", "concept": "ATP", "question": "?", "options": ["a", "b", "c", "d"], "difficulty": 0.5, "kind": "practice"}

# (client call, REST (method, path, query, json body), RPC (method, params), the server's result, check on the parsed result)
CASES = [
    (lambda c: c.get_structure("d1"), ("GET", "/v1/documents/d1/structure", {}, None), ("document.structure", {"doc_id": "d1"}),
     {"doc_id": "d1", "title": "T", "format": "pdf", "pages": 3, "sections": [{"id": "s1", "ordinal": 1, "title": "A", "depth": 1, "page_start": 1, "page_end": 2}]},
     lambda r: r.sections[0].title == "A" and r.pages == 3),
    (lambda c: c.get_knowledge("d1", upto=40), ("GET", "/v1/documents/d1/knowledge", {"upto": "40"}, None), ("document.knowledge", {"doc_id": "d1", "upto": 40}),
     {"as_of": 40, "concepts": [CONCEPT], "relations": [{"src": "c1", "dst": "c2", "type": "prerequisite_of", "strength": 2}]},
     lambda r: r.as_of == 40 and r.concepts[0].name == "ATP" and r.relations[0].type == "prerequisite_of"),
    (lambda c: c.get_knowledge("d1"), ("GET", "/v1/documents/d1/knowledge", {}, None), ("document.knowledge", {"doc_id": "d1"}),
     {"as_of": None, "concepts": [], "relations": []}, lambda r: r.as_of is None),
    (lambda c: c.get_section_knowledge("d1", 7), ("GET", "/v1/documents/d1/sections/7/knowledge", {}, None),
     ("document.section_knowledge", {"doc_id": "d1", "section_id": 7}),
     {"section_id": "s1.2", "title": "Calvin", "introduces": [CONCEPT], "uses_earlier": [],
      "depends_on": [{"section_id": "s1.1", "title": "Light", "concepts": ["ATP"], "weight": 1}]},
     lambda r: r.depends_on[0].title == "Light" and r.introduces[0].id == "c1"),
    (lambda c: c.get_growth("d1"), ("GET", "/v1/documents/d1/growth", {}, None), ("document.growth", {"doc_id": "d1"}),
     {"points": [{"step": 0, "section_id": "s1", "ordinal_end": 4, "words": 50, "new_concepts": 2, "total_concepts": 2, "total_relations": 0, "novelty": 40.0, "reuse": 0.0}]},
     lambda r: r[0].total_concepts == 2),
    (lambda c: c.get_prerequisites("d1", "cellular respiration", max_depth=2), ("GET", "/v1/documents/d1/concepts/cellular%20respiration/prerequisites", {"max_depth": "2"}, None),
     ("concept.prerequisites", {"doc_id": "d1", "name": "cellular respiration", "max_depth": 2}), {"concept": "cellular respiration", "data": [{"name": "sugar"}]},
     lambda r: r.data[0]["name"] == "sugar"),
    (lambda c: c.get_learning_path("d1", "x"), ("GET", "/v1/documents/d1/concepts/x/learning-path", {}, None), ("concept.learning_path", {"doc_id": "d1", "name": "x"}),
     {"concept": "x", "data": []}, lambda r: r.concept == "x"),
    (lambda c: c.recommend_concepts("d1", "x", limit=3), ("GET", "/v1/documents/d1/concepts/x/recommendations", {"limit": "3"}, None),
     ("concept.recommendations", {"doc_id": "d1", "name": "x", "limit": 3}), {"concept": "x", "data": []}, lambda r: r.concept == "x"),
    (lambda c: c.get_learner("d1"), ("GET", "/v1/documents/d1/learner", {}, None), ("learner.get", {"doc_id": "d1"}),
     {"level": "beginner", "prior": 0.15, "position": None, "concepts": [{"concept_id": "c1", "name": "ATP", "p_known": 0.2, "evidence": 1}]},
     lambda r: r.level == "beginner" and r.concepts[0].p_known == 0.2 and r.position is None),
    (lambda c: c.start_diagnostic("d1", "beginner", ["algebra"], 4), ("POST", "/v1/documents/d1/diagnostic", {}, {"level": "beginner", "known_topics": ["algebra"], "questions": 4}),
     ("learner.diagnostic_start", {"doc_id": "d1", "level": "beginner", "known_topics": ["algebra"], "questions": 4}),
     {"level": "beginner", "prior": 0.2, "questions": [QUESTION]}, lambda r: r.questions[0].id == "q1"),
    (lambda c: c.finish_diagnostic("d1"), ("POST", "/v1/documents/d1/diagnostic/finish", {}, None), ("learner.diagnostic_finish", {"doc_id": "d1"}),
     {"level": "advanced", "prior": 0.7, "questions": []}, lambda r: r.level == "advanced"),
    (lambda c: c.practice_quiz("d1", section_id=9, count=2), ("POST", "/v1/documents/d1/quiz", {}, {"section_id": 9, "count": 2}),
     ("quiz.new", {"doc_id": "d1", "section_id": 9, "count": 2}), {"questions": [QUESTION]}, lambda r: r[0].options == ["a", "b", "c", "d"]),
    (lambda c: c.practice_quiz("d1"), ("POST", "/v1/documents/d1/quiz", {}, {"count": 5}), ("quiz.new", {"doc_id": "d1", "count": 5}),
     {"questions": []}, lambda r: r == []),
    (lambda c: c.answer_quiz("d1", "q1", 2), ("POST", "/v1/documents/d1/quiz/q1/answer", {}, {"choice": 2}), ("quiz.answer", {"doc_id": "d1", "question_id": "q1", "choice": 2}),
     {"correct": True, "correct_index": 2, "explanation": "because", "already_answered": False}, lambda r: r.correct and r.correct_index == 2),
    (lambda c: c.send_feedback("d1", "too_hard", ["c1"]), ("POST", "/v1/documents/d1/feedback", {}, {"verdict": "too_hard", "concept_ids": ["c1"]}),
     ("learner.feedback", {"doc_id": "d1", "verdict": "too_hard", "concept_ids": ["c1"]}), {"ok": True}, lambda r: r is None),
    (lambda c: c.mark_progress("d1", 12), ("POST", "/v1/documents/d1/progress", {}, {"section_id": 12}), ("learner.progress", {"doc_id": "d1", "section_id": 12}),
     {"ok": True}, lambda r: r is None),
]


@pytest.mark.parametrize("call,rest,rpc_,result,check", CASES, ids=[str(i) for i in range(len(CASES))])
def test_every_knowledge_and_learner_operation_is_correct_on_both_protocols(call, rest, rpc_, result, check):
    seen = {}

    def rest_handler(req):
        seen["rest"] = (req.method, req.url.path, dict(req.url.params), json.loads(req.content) if req.content else None)
        return httpx.Response(200, json=result)

    assert check(call(make(rest_handler)))
    method, path, query, body = rest
    assert seen["rest"][0] == method and seen["rest"][1].replace("%20", " ") == path.replace("%20", " ")
    assert seen["rest"][2] == query and seen["rest"][3] == body

    def rpc_handler(req):
        b = json.loads(req.content)
        seen["rpc"] = (b["method"], b["params"])
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": result, "id": b["id"]})

    assert check(call(make(rpc_handler, protocol="rpc")))
    assert seen["rpc"] == rpc_


def test_deleting_a_document_works_on_both_protocols():
    seen = []
    c = make(lambda req: (seen.append((req.method, req.url.path)), httpx.Response(204))[1])
    assert c.delete_document("d1") is None and seen == [("DELETE", "/v1/documents/d1")]
    r = make(lambda req: httpx.Response(200, json={"jsonrpc": "2.0", "result": {"ok": True}, "id": json.loads(req.content)["id"]}), protocol="rpc")
    assert r.delete_document("d1") is None


def test_not_found_codes_for_questions_and_concepts_map_to_NotFoundError():
    for code in ("question_not_found", "concept_not_found"):
        h = lambda req, c=code: httpx.Response(404, json={"error": {"code": c, "message": "no"}})
        with pytest.raises(sdk.NotFoundError) as e:
            make(h).answer_quiz("d1", "q", 0)
        assert e.value.code == code
    rpc_h = lambda req: httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32011, "message": "no concept"}})
    with pytest.raises(sdk.NotFoundError):
        make(rpc_h, protocol="rpc").get_prerequisites("d1", "zzz")


async def test_the_async_client_has_every_method_the_sync_one_has():
    sync = {n for n in dir(sdk.DocExplainerClient) if not n.startswith("_")}
    asynchronous = {n for n in dir(sdk.AsyncDocExplainerClient) if not n.startswith("_")}
    assert sync - {"close"} == asynchronous - {"aclose"}
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=CASES[0][3])))
    async with sdk.AsyncDocExplainerClient("http://srv", http_client=http) as c:
        assert (await c.get_structure("d1")).sections[0].title == "A"
