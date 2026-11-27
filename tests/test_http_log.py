"""The server's HTTP log: one line per request in http_logs.log, no bodies/headers/queries, nothing on the console."""
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from doc_explainer.api.http_log import HTTP_LOGGER, setup_http_logging
from doc_explainer.config import BackendConfig

from fakes import Brain
from test_api import _close_services, doc_id, make_client  # noqa: F401  (fixture re-used)


@pytest.fixture
def log_file(tmp_path):
    f = tmp_path / "logs" / "http_logs.log"
    setup_http_logging(f)
    yield f
    logging.getLogger(HTTP_LOGGER).handlers.clear()
    logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())


def lines(f):
    for h in logging.getLogger(HTTP_LOGGER).handlers:
        h.flush()
    return f.read_text().splitlines()


def test_each_request_is_one_line_with_method_path_status_time_and_size(tmp_path, log_file):
    c = make_client(tmp_path, Brain())
    c.get("/v1/health")
    c.get("/v1/documents/nope")
    out = lines(log_file)
    assert len(out) == 2
    assert "GET /v1/health -> 200 (" in out[0] and " ms, sent " in out[0] and out[0].endswith(" B)")
    assert "GET /v1/documents/nope -> 404" in out[1]


def test_validation_errors_unauthorised_and_unknown_routes_are_logged_with_their_status(tmp_path, log_file):
    c = make_client(tmp_path, Brain(), api_key="secret")
    c.get("/v1/documents")                                                                       # no key
    c.get("/v1/nothing-here")
    c.post("/v1/documents/x/ask", json={}, headers={"Authorization": "Bearer secret"})
    out = " | ".join(lines(log_file))
    assert "GET /v1/documents -> 401" in out and "GET /v1/nothing-here -> 404" in out and "POST /v1/documents/x/ask -> 422" in out


def test_secrets_questions_queries_and_bodies_never_reach_the_log(tmp_path, log_file):
    c = make_client(tmp_path, Brain(), api_key="super-secret-key")
    h = {"Authorization": "Bearer super-secret-key", "X-User-Id": "alice"}
    d = doc_id(c, headers=h)
    c.post(f"/v1/documents/{d}/ask", json={"question": "my private question about salaries"}, headers=h)
    c.get(f"/v1/documents/{d}/knowledge?upto=3", headers=h)
    text = log_file.read_text()
    assert f"POST /v1/documents/{d}/ask -> 200" in text
    for leaked in ("super-secret-key", "private question", "salaries", "Authorization", "alice", "upto=3", "Photosynthesis"):
        assert leaked not in text


def test_a_streamed_reply_is_logged_when_it_ends_with_the_whole_size(tmp_path, log_file):
    brain = Brain()
    c = make_client(tmp_path, brain)
    d = doc_id(c)
    r = c.post(f"/v1/documents/{d}/chat/stream", json={"kind": "ask", "text": "hi"})
    line = next(l for l in lines(log_file) if "/chat/stream" in l)
    assert "-> 200" in line and f"sent {len(r.content)} B" in line                                 # measured to the last byte


def test_an_unexpected_crash_is_logged_as_a_failure_and_still_raises(tmp_path, log_file):
    c = make_client(tmp_path, Brain())
    c.app.state.service.list_documents = lambda user: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        TestClient(c.app, raise_server_exceptions=True).get("/v1/documents")
    assert any("GET /v1/documents -> FAILED RuntimeError" in l for l in lines(log_file))


def test_the_http_log_stays_out_of_the_console_and_the_application_log(tmp_path, log_file):
    seen = []

    class Spy(logging.Handler):
        def emit(self, record):
            seen.append(record)
    spy = Spy(level=logging.DEBUG)
    logging.getLogger().addHandler(spy)
    try:
        make_client(tmp_path, Brain()).get("/v1/health")
    finally:
        logging.getLogger().removeHandler(spy)
    assert [r for r in seen if r.name == HTTP_LOGGER] == [] and logging.getLogger(HTTP_LOGGER).propagate is False
    assert logging.getLogger("uvicorn.access").propagate is False


def test_without_setup_nothing_is_written_or_raised(tmp_path):
    logging.getLogger(HTTP_LOGGER).handlers.clear()
    logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())
    assert make_client(tmp_path, Brain()).get("/v1/health").status_code == 200


def test_the_file_name_comes_from_the_config_and_the_file_rotates(tmp_path, monkeypatch):
    cfg = BackendConfig()
    cfg.logging.log_directory = str(tmp_path / "logs")
    assert cfg.http_log_file == tmp_path / "logs" / "http_logs.log"
    monkeypatch.setenv("DOC_EXPLAINER_LOGGING_HTTP_LOG_FILE", "requests.log")
    assert BackendConfig.load().logging.http_log_file == "requests.log"
    f = tmp_path / "r.log"
    setup_http_logging(f, max_bytes=300, backups=2)
    c = make_client(tmp_path, Brain())
    for _ in range(25):
        c.get("/v1/health")
    for h in logging.getLogger(HTTP_LOGGER).handlers:
        h.flush()
    assert (tmp_path / "r.log.1").exists()
    logging.getLogger(HTTP_LOGGER).handlers.clear()
    logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())


def test_the_server_starts_uvicorn_without_its_own_access_log(monkeypatch):
    import uvicorn
    from doc_explainer.api import cli
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["doc-explainer-server"])
    cli.main([])
    assert captured["access_log"] is False
