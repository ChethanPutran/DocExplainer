"""Errors raised by the service layer; REST and RPC each translate them into their own wire format."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    http_status: int
    rpc_code: int


# One table: every stable error code the API can return, with its REST status and JSON-RPC code.
ERRORS = {
    "invalid_params":         ErrorSpec(422, -32602),
    "unauthorized":           ErrorSpec(401, -32001),
    "processing_failed":      ErrorSpec(422, -32002),
    "not_ready":              ErrorSpec(503, -32003),
    "document_not_found":     ErrorSpec(404, -32004),
    "file_too_large":         ErrorSpec(413, -32005),
    "unsupported_file_type":  ErrorSpec(415, -32006),
    "internal_error":         ErrorSpec(500, -32603),
    "job_not_found":          ErrorSpec(404, -32008),
    "queue_full":             ErrorSpec(429, -32009),
    "question_not_found":     ErrorSpec(404, -32010),
    "concept_not_found":      ErrorSpec(404, -32011),
    "session_not_found":      ErrorSpec(404, -32012),
    "note_not_found":         ErrorSpec(404, -32013),
    "job_not_active":         ErrorSpec(409, -32014),
    "nothing_to_summarize":   ErrorSpec(409, -32015),
    "empty_summary":          ErrorSpec(422, -32016),
}


class ServiceError(Exception):
    """A failure the client should be told about, with a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str):
        if code not in ERRORS:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return ERRORS[self.code].http_status

    @property
    def rpc_code(self) -> int:
        return ERRORS[self.code].rpc_code
