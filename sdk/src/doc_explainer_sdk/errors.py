"""Exceptions raised by the SDK. Both protocols raise the same ones for the same failure."""
from __future__ import annotations
from typing import Any, Optional


class DocExplainerError(Exception):
    """Base class. ``code`` is the server's stable error code (or an SDK-side one)."""

    def __init__(self, message: str, *, code: str = "error", status: Optional[int] = None,
                 details: Any = None):
        super().__init__(message)
        self.message, self.code, self.status, self.details = message, code, status, details

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ServerUnavailableError(DocExplainerError):
    """Could not reach the server, it timed out, or it is still starting. Safe to retry later."""


class JobCancelledError(DocExplainerError):
    """The job was cancelled (by you or someone using your account), so there is no document."""


class ServerBusyError(ServerUnavailableError):
    """The server's processing queue is full. Retry shortly."""


class JobTimeoutError(DocExplainerError):
    """``wait_for_job`` gave up waiting. The job is still running on the server: wait again with
    ``get_job`` / ``wait_for_job(job_id)`` using ``.job_id``."""

    def __init__(self, message: str, *, job_id: str, job: Any = None):
        super().__init__(message, code="job_timeout")
        self.job_id, self.job = job_id, job


class AuthenticationError(DocExplainerError):
    """Missing or invalid API key."""


class NotFoundError(DocExplainerError):
    """The document does not exist (or belongs to another user)."""


class InvalidRequestError(DocExplainerError):
    """The request was rejected: bad parameters, unsupported file type, file too large."""


class ProcessingError(DocExplainerError):
    """The server understood the request but could not complete it (e.g. the model failed)."""


class ServerError(DocExplainerError):
    """An unexpected failure on the server."""


class ProtocolError(DocExplainerError):
    """The server's reply was not what the protocol allows (wrong server, proxy page, ...)."""


_BY_CODE = {
    "unauthorized": AuthenticationError,
    "document_not_found": NotFoundError,
    "invalid_params": InvalidRequestError,
    "unsupported_file_type": InvalidRequestError,
    "file_too_large": InvalidRequestError,
    "processing_failed": ProcessingError,
    "not_ready": ServerUnavailableError,
    "queue_full": ServerBusyError,
    "job_not_found": NotFoundError,
    "question_not_found": NotFoundError,
    "concept_not_found": NotFoundError,
    "session_not_found": NotFoundError,
    "note_not_found": NotFoundError,
    "job_not_active": InvalidRequestError,
    "nothing_to_summarize": InvalidRequestError,
    "empty_summary": ProcessingError,
    "internal_error": ServerError,
}


def error_from_code(code: str, message: str, *, status: Optional[int] = None,
                    details: Any = None) -> DocExplainerError:
    return _BY_CODE.get(code, ServerError)(message, code=code, status=status, details=details)
