"""All HTTP the UI sends is logged to http_logs.log: one line each, never bodies or headers."""
import logging

import httpx
import pytest

from doc_explainer_sdk import DocExplainerClient
from doc_explainer_ui.gui.utils.http_log import HTTP_LOGGER, LoggingTransport, setup_http_logging

HEALTH = {"status": "ok", "version": "1", "api_version": "v1"}


@pytest.fixture
def log_file(tmp_path):
    f = tmp_path / "logs" / "http_logs.log"
    setup_http_logging(f)
    yield f
    setup_http_logging(tmp_path / "x.log")          # leave a handler that is closed with tmp_path
    logging.getLogger(HTTP_LOGGER).handlers.clear()


def client(handler, protocol="rest", **kw):
    http = httpx.Client(transport=LoggingTransport(httpx.MockTransport(handler)))
    return DocExplainerClient("http://srv", protocol=protocol, http_client=http, **kw)


def lines(f):
    for h in logging.getLogger(HTTP_LOGGER).handlers:
        h.flush()
    return f.read_text().splitlines()


def test_a_rest_request_is_one_line_with_method_path_status_and_time(log_file):
    client(lambda r: httpx.Response(200, json=HEALTH)).health()
    (line,) = lines(log_file)
    assert "GET /v1/health -> 200 OK" in line and " ms" in line


def test_a_json_rpc_request_shows_the_rpc_method(log_file):
    client(lambda r: httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": HEALTH}), "rpc").health()
    assert "POST /rpc rpc=system.health -> 200" in lines(log_file)[0]


def test_bodies_headers_and_the_api_key_are_never_written(log_file):
    seen = []
    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"text": "answer", "explanation": {"explanation": "secret answer"}})
    c = client(handler, api_key="super-secret-key", user_id="alice")
    c.ask("d1", "my private question about salaries")
    text = log_file.read_text()
    assert "POST /v1/documents/d1/ask -> 200" in text
    for leaked in ("super-secret-key", "private question", "salaries", "secret answer", "Authorization", "alice"):
        assert leaked not in text


def test_a_failed_connection_is_logged_as_a_warning_with_the_reason(log_file):
    def refuse(req):
        raise httpx.ConnectError("connection refused")
    with pytest.raises(Exception):
        client(refuse).health()
    line = lines(log_file)[0]
    assert "GET /v1/health -> FAILED ConnectError: connection refused" in line


def test_http_errors_are_logged_with_their_status(log_file):
    with pytest.raises(Exception):
        client(lambda r: httpx.Response(404, json={"error": {"code": "document_not_found", "message": "x"}})).get_document("zz")
    assert "-> 404" in lines(log_file)[0]


def test_every_request_gets_its_own_line_in_order(log_file):
    c = client(lambda r: httpx.Response(200, json=HEALTH))
    c.health(); c.health(); c.health()
    assert len(lines(log_file)) == 3


def test_the_http_log_stays_out_of_the_console_and_the_app_log(log_file, caplog):
    root_hits = []
    class Spy(logging.Handler):
        def emit(self, record):
            root_hits.append(record)
    spy = Spy(level=logging.DEBUG)
    logging.getLogger().addHandler(spy)
    try:
        client(lambda r: httpx.Response(200, json=HEALTH)).health()
    finally:
        logging.getLogger().removeHandler(spy)
    assert root_hits == [] and logging.getLogger(HTTP_LOGGER).propagate is False
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_without_setup_nothing_is_written_or_raised(tmp_path):
    logging.getLogger(HTTP_LOGGER).handlers.clear()
    logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())
    assert client(lambda r: httpx.Response(200, json=HEALTH)).health().status == "ok"


def test_the_log_file_rotates(tmp_path):
    f = tmp_path / "r.log"
    setup_http_logging(f, max_bytes=300, backups=2)
    c = client(lambda r: httpx.Response(200, json=HEALTH))
    for _ in range(30):
        c.health()
    for h in logging.getLogger(HTTP_LOGGER).handlers:
        h.flush()
    assert (tmp_path / "r.log.1").exists() and f.stat().st_size <= 600
    logging.getLogger(HTTP_LOGGER).handlers.clear()
