"""The two wire protocols. Both are driven by the same ``OPERATIONS`` table, so a REST call and an
RPC call for the same operation take the same logical parameters and return the same result.

A protocol only *builds* a request and *parses* a response; the client sends it (sync or async)."""
from __future__ import annotations
import base64
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import httpx

from .errors import (DocExplainerError, ProtocolError, ServerError, ServerUnavailableError,
                      AuthenticationError, error_from_code)

PROTOCOLS = ("rest", "rpc")


@dataclass(frozen=True)
class Operation:
    rpc: str                       # JSON-RPC method name
    method: str                    # REST verb
    path: str                      # REST path; {name} placeholders come from the params
    query: Tuple[str, ...] = ()    # REST: params sent in the query string
    body: Tuple[str, ...] = ()     # REST: params sent as the JSON body
    upload: bool = False           # REST: multipart file upload


OPERATIONS: Dict[str, Operation] = {
    "health":          Operation("system.health", "GET", "/v1/health"),
    "register":        Operation("document.register", "POST", "/v1/documents", upload=True),
    "list_documents":  Operation("document.list", "GET", "/v1/documents"),
    "get_job":         Operation("job.get", "GET", "/v1/jobs/{job_id}"),
    "list_jobs":       Operation("job.list", "GET", "/v1/jobs"),
    "get_document":    Operation("document.get", "GET", "/v1/documents/{doc_id}"),
    "section_at":      Operation("document.section_at", "GET", "/v1/documents/{doc_id}/section",
                                 query=("page", "position")),
    "context":         Operation("document.context", "GET", "/v1/documents/{doc_id}/context",
                                 query=("section_id",)),
    "explain":         Operation("document.explain", "POST", "/v1/documents/{doc_id}/explain",
                                 body=("selected_text", "section_id", "style")),
    "summarize":       Operation("document.summarize", "POST", "/v1/documents/{doc_id}/summarize",
                                 body=("selected_text", "section_id")),
    "ask":             Operation("document.ask", "POST", "/v1/documents/{doc_id}/ask",
                                 body=("question", "section_id")),
    "delete_document": Operation("document.delete", "DELETE", "/v1/documents/{doc_id}"),
    "structure":       Operation("document.structure", "GET", "/v1/documents/{doc_id}/structure"),
    "knowledge":       Operation("document.knowledge", "GET", "/v1/documents/{doc_id}/knowledge", query=("upto",)),
    "section_knowledge": Operation("document.section_knowledge", "GET", "/v1/documents/{doc_id}/sections/{section_id}/knowledge"),
    "delete_knowledge": Operation("document.delete_knowledge", "DELETE", "/v1/documents/{doc_id}/knowledge"),
    "review":          Operation("learner.review", "GET", "/v1/documents/{doc_id}/review", query=("limit",)),
    "history":         Operation("learner.history", "GET", "/v1/documents/{doc_id}/history", query=("limit",)),
    "export_session":  Operation("session.export", "GET", "/v1/documents/{doc_id}/sessions/{session_id}/export"),
    "growth":          Operation("document.growth", "GET", "/v1/documents/{doc_id}/growth"),
    "prerequisites":   Operation("concept.prerequisites", "GET", "/v1/documents/{doc_id}/concepts/{name}/prerequisites",
                                 query=("max_depth",)),
    "learning_path":   Operation("concept.learning_path", "GET", "/v1/documents/{doc_id}/concepts/{name}/learning-path"),
    "recommendations": Operation("concept.recommendations", "GET", "/v1/documents/{doc_id}/concepts/{name}/recommendations",
                                 query=("limit",)),
    "learner":         Operation("learner.get", "GET", "/v1/documents/{doc_id}/learner"),
    "diagnostic_start": Operation("learner.diagnostic_start", "POST", "/v1/documents/{doc_id}/diagnostic",
                                  body=("level", "known_topics", "questions")),
    "diagnostic_finish": Operation("learner.diagnostic_finish", "POST", "/v1/documents/{doc_id}/diagnostic/finish"),
    "quiz_new":        Operation("quiz.new", "POST", "/v1/documents/{doc_id}/quiz", body=("section_id", "count")),
    "quiz_answer":     Operation("quiz.answer", "POST", "/v1/documents/{doc_id}/quiz/{question_id}/answer", body=("choice",)),
    "feedback":        Operation("learner.feedback", "POST", "/v1/documents/{doc_id}/feedback", body=("verdict", "concept_ids")),
    "progress":        Operation("learner.progress", "POST", "/v1/documents/{doc_id}/progress", body=("section_id",)),
    "pause_job":       Operation("job.pause", "POST", "/v1/jobs/{job_id}/pause"),
    "resume_job":      Operation("job.resume", "POST", "/v1/jobs/{job_id}/resume"),
    "cancel_job":      Operation("job.cancel", "POST", "/v1/jobs/{job_id}/cancel"),
    "chat":            Operation("chat.send", "POST", "/v1/documents/{doc_id}/chat", body=("kind", "text", "section_id", "session_id", "level")),
    # streamed over REST only (JSON-RPC is request/response): the client replays a non-streamed turn instead
    "chat_stream":     Operation("chat.send", "POST", "/v1/documents/{doc_id}/chat/stream", body=("kind", "text", "section_id", "session_id", "level")),
    "list_sessions":   Operation("session.list", "GET", "/v1/documents/{doc_id}/sessions"),
    "create_session":  Operation("session.create", "POST", "/v1/documents/{doc_id}/sessions", body=("title",)),
    "get_session":     Operation("session.get", "GET", "/v1/documents/{doc_id}/sessions/{session_id}"),
    "rename_session":  Operation("session.rename", "PATCH", "/v1/documents/{doc_id}/sessions/{session_id}", body=("title",)),
    "delete_session":  Operation("session.delete", "DELETE", "/v1/documents/{doc_id}/sessions/{session_id}"),
    "summarize_session": Operation("session.summarize", "POST", "/v1/documents/{doc_id}/sessions/{session_id}/summary"),
    "add_note":        Operation("note.add", "POST", "/v1/documents/{doc_id}/sessions/{session_id}/notes", body=("text", "kind")),
    "edit_note":       Operation("note.edit", "PUT", "/v1/documents/{doc_id}/sessions/{session_id}/notes/{note_id}", body=("text",)),
    "delete_note":     Operation("note.delete", "DELETE", "/v1/documents/{doc_id}/sessions/{session_id}/notes/{note_id}"),
}


@dataclass
class RequestSpec:
    method: str
    url: str
    params: Optional[dict] = None
    json: Any = None
    data: Optional[dict] = None
    files: Optional[dict] = None
    request_id: Any = None


def _clean(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _status_error(status: int, message: str) -> DocExplainerError:
    if status in (502, 503, 504):
        return ServerUnavailableError(message, code="unavailable", status=status)
    if status == 401:
        return AuthenticationError(message, code="unauthorized", status=status)
    return ServerError(message, code="http_error", status=status)


class RestProtocol:
    name = "rest"

    def __init__(self, base_url: str):
        self._base = base_url

    def build(self, op: Operation, params: Dict[str, Any]) -> RequestSpec:
        params = _clean(params)
        path = op.path.format(**{k: quote(str(v), safe="") for k, v in params.items()})
        spec = RequestSpec(op.method, self._base + path)
        query = {k: params[k] for k in op.query if k in params}
        if query:
            spec.params = query
        if op.upload:
            spec.files = {"file": (params["filename"], params["content"])}
            spec.data = {"build_graph": "true" if params.get("build_graph", True) else "false"}
        elif op.body:
            spec.json = {k: params[k] for k in op.body if k in params}
        return spec

    def parse(self, response: httpx.Response, request_id: Any = None) -> Any:
        if response.is_success:
            if response.status_code == 204 or not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                raise ProtocolError("The server returned a success status with a non-JSON body.",
                                    code="protocol_error", status=response.status_code)
        try:
            err = response.json()["error"]
            code, message = str(err["code"]), str(err["message"])
        except (ValueError, KeyError, TypeError):
            raise _status_error(response.status_code,
                                f"HTTP {response.status_code} from the server (not a Doc Explainer "
                                f"error response; is the URL right?)")
        raise error_from_code(code, message, status=response.status_code, details=err.get("details"))


_RPC_TO_CODE = {-32602: "invalid_params", -32001: "unauthorized", -32002: "processing_failed",
                -32003: "not_ready", -32004: "document_not_found", -32005: "file_too_large",
                -32006: "unsupported_file_type", -32008: "job_not_found", -32009: "queue_full",
                -32010: "question_not_found", -32011: "concept_not_found", -32012: "session_not_found",
                -32013: "note_not_found", -32014: "job_not_active", -32015: "nothing_to_summarize",
                -32016: "empty_summary", -32603: "internal_error"}


class RpcProtocol:
    name = "rpc"

    def __init__(self, base_url: str):
        self._url = base_url + "/rpc"
        self._ids = itertools.count(1)

    def build(self, op: Operation, params: Dict[str, Any]) -> RequestSpec:
        params = _clean(params)
        if op.upload:
            params = {"filename": params["filename"], "build_graph": params.get("build_graph", True),
                      "content_b64": base64.b64encode(params["content"]).decode("ascii")}
        rid = next(self._ids)
        return RequestSpec("POST", self._url, request_id=rid,
                           json={"jsonrpc": "2.0", "method": op.rpc, "params": params, "id": rid})

    def parse(self, response: httpx.Response, request_id: Any = None) -> Any:
        # Auth and size problems are rejected before JSON-RPC dispatch, as plain HTTP errors.
        if response.status_code in (401, 413) or (not response.is_success):
            try:
                err = response.json()["error"]
                raise error_from_code(str(err["code"]), str(err["message"]), status=response.status_code)
            except (ValueError, KeyError, TypeError):
                raise _status_error(response.status_code, f"HTTP {response.status_code} from the server")
        try:
            msg = response.json()
        except ValueError:
            raise ProtocolError("The server returned a non-JSON body.", code="protocol_error",
                                status=response.status_code)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            raise ProtocolError("Not a JSON-RPC 2.0 response; is the URL right?", code="protocol_error")
        if "error" in msg:
            e = msg["error"] or {}
            data = e.get("data") or {}
            code = data.get("code") or _RPC_TO_CODE.get(e.get("code"), "internal_error")
            raise error_from_code(code, str(e.get("message", "RPC error")),
                                  status=response.status_code, details=data.get("details"))
        if request_id is not None and msg.get("id") != request_id:
            raise ProtocolError("Response id does not match the request id.", code="protocol_error")
        return msg.get("result")


def make_protocol(name: str, base_url: str):
    if name == "rest":
        return RestProtocol(base_url)
    if name == "rpc":
        return RpcProtocol(base_url)
    raise ValueError(f"protocol must be one of {PROTOCOLS}, got {name!r}")
