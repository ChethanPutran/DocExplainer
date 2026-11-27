"""Every HTTP request the UI sends to the service goes to ``http_logs.log`` (and nowhere else).

One line per request: method, path, (the JSON-RPC method), status, time, sizes. Request and response
BODIES and HEADERS are never logged: they hold the API key, the user's questions and the document text.
"""
from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import httpx

HTTP_LOGGER = "doc_explainer_ui.http"
logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())      # silent unless setup_http_logging() ran


def setup_http_logging(path: Path, max_bytes: int = 5_000_000, backups: int = 3) -> logging.Logger:
    """Send the HTTP log to ``path`` only: not to the console, not to app.log."""
    logger = logging.getLogger(HTTP_LOGGER)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # httpx logs its own "HTTP Request: ..." line; keep that out of the console/app.log too.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logger


def _rpc_method(request: httpx.Request) -> str:
    if request.url.path.rstrip("/") != "/rpc":
        return ""
    try:
        return " rpc=" + str(json.loads(request.content).get("method", ""))
    except Exception:  # noqa: BLE001  (a body we cannot read is simply not described)
        return ""


class LoggingTransport(httpx.BaseTransport):
    """Wraps a real transport and logs each request once it has finished (or failed)."""

    def __init__(self, inner: Optional[httpx.BaseTransport] = None):
        self._inner = inner or httpx.HTTPTransport()
        self._log = logging.getLogger(HTTP_LOGGER)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        where = f"{request.method} {request.url.raw_path.decode()}{_rpc_method(request)}"
        sent = request.headers.get("content-length", "0")
        try:
            response = self._inner.handle_request(request)
        except Exception as e:  # noqa: BLE001
            self._log.warning("%s -> FAILED %s: %s (%.0f ms)", where, type(e).__name__, e, (time.monotonic() - started) * 1000)
            raise
        self._log.info("%s -> %s %s (%.0f ms, sent %s B, got %s B)", where, response.status_code, response.reason_phrase,
                       (time.monotonic() - started) * 1000, sent, response.headers.get("content-length", "?"))
        return response

    def close(self) -> None:
        self._inner.close()


def logging_http_client() -> httpx.Client:
    """The HTTP client the UI hands to the SDK."""
    return httpx.Client(transport=LoggingTransport())
