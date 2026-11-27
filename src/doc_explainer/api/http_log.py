"""Every HTTP request the server handles goes to ``http_logs.log`` (and nowhere else).

One line per request: method, path, status, time, bytes sent. Query strings, headers (the API key!), request
bodies and response bodies are never logged: they hold the reader's questions and the document text. uvicorn's own
access log is switched off (``access_log=False``) so a request is not reported twice, or onto the console.
"""
from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

HTTP_LOGGER = "doc_explainer.http"
logging.getLogger(HTTP_LOGGER).addHandler(logging.NullHandler())          # silent unless setup_http_logging() ran


def setup_http_logging(path: Path, max_bytes: int = 5_000_000, backups: int = 3) -> logging.Logger:
    """Send the HTTP log to ``path`` only: not to the console, not to the application log."""
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
    for name in ("uvicorn.access",):                      # in case the server was started some other way
        access = logging.getLogger(name)
        access.handlers.clear()
        access.propagate = False
    return logger


class HttpLogMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware) so streamed replies are measured to their last byte, not to their first."""

    def __init__(self, app):
        self.app = app
        self.log = logging.getLogger(HTTP_LOGGER)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        state = {"status": 0, "bytes": 0}

        async def logged_send(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
            elif message["type"] == "http.response.body":
                state["bytes"] += len(message.get("body", b""))
            await send(message)

        where = f"{scope['method']} {scope['path']}"
        try:
            await self.app(scope, receive, logged_send)
        except Exception as e:  # noqa: BLE001 - logged, then left for the framework to turn into a 500
            self.log.warning("%s -> FAILED %s (%.0f ms)", where, type(e).__name__, (time.monotonic() - started) * 1000)
            raise
        self.log.info("%s -> %s (%.0f ms, sent %s B)", where, state["status"], (time.monotonic() - started) * 1000, state["bytes"])
