"""RPC protocol: JSON-RPC 2.0 over ``POST /rpc`` (same port and auth as REST, no code generation).

Supports named params, notifications (no ``id`` -> no response) and batches. Methods are declared
once in ``METHODS`` and call the same service methods as the REST routes.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from .errors import ServiceError
from .rest import authorize, current_user, get_service
from .schemas import (
    Ack, ChatParams, NoteEditParams, NoteIdParams, NoteParams, SessionCreateParams, SessionIdParams, SessionRenameParams,
    AnswerParams, AskParams, ConceptParams, ContextParams, DiagnosticParams, DocIdParams, ExplainParams, FeedbackParams,
    JobIdParams, KnowledgeParams, LearningPathParams, NoParams, PrerequisiteParams, ProgressParams, QuizParams, RecommendParams,
    LimitParams, RegisterParams, SectionAtParams, SectionParams, SummarizeParams,
)
from .service import DocumentApiService

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL = -32700, -32600, -32601, -32602, -32603
MAX_BATCH = 20


@dataclass(frozen=True)
class RpcMethod:
    params: Type[BaseModel]
    call: Callable[[DocumentApiService, str, Any], Any]


METHODS: Dict[str, RpcMethod] = {
    "system.health":        RpcMethod(NoParams, lambda s, u, p: s.health()),
    "document.register":    RpcMethod(RegisterParams, lambda s, u, p: s.register_document_b64(
                                u, p.filename, p.content_b64, p.build_graph)),
    "document.list":        RpcMethod(NoParams, lambda s, u, p: s.list_documents(u)),
    "job.get":              RpcMethod(JobIdParams, lambda s, u, p: s.get_job(u, p.job_id)),
    "job.list":             RpcMethod(NoParams, lambda s, u, p: s.list_jobs(u)),
    "job.pause":            RpcMethod(JobIdParams, lambda s, u, p: s.pause_job(u, p.job_id)),
    "job.resume":           RpcMethod(JobIdParams, lambda s, u, p: s.resume_job(u, p.job_id)),
    "job.cancel":           RpcMethod(JobIdParams, lambda s, u, p: s.cancel_job(u, p.job_id)),
    "chat.send":            RpcMethod(ChatParams, lambda s, u, p: s.chat_once(u, p.doc_id, p)),
    "session.list":         RpcMethod(DocIdParams, lambda s, u, p: s.list_sessions(u, p.doc_id)),
    "session.create":       RpcMethod(SessionCreateParams, lambda s, u, p: s.create_session(u, p.doc_id, p.title)),
    "session.get":          RpcMethod(SessionIdParams, lambda s, u, p: s.get_session(u, p.doc_id, p.session_id)),
    "session.rename":       RpcMethod(SessionRenameParams, lambda s, u, p: s.rename_session(u, p.doc_id, p.session_id, p.title)),
    "session.delete":       RpcMethod(SessionIdParams, lambda s, u, p: (s.delete_session(u, p.doc_id, p.session_id), Ack())[1]),
    "session.summarize":    RpcMethod(SessionIdParams, lambda s, u, p: s.summarize_session(u, p.doc_id, p.session_id)),
    "note.add":             RpcMethod(NoteParams, lambda s, u, p: s.add_note(u, p.doc_id, p.session_id, p.text, p.kind)),
    "note.edit":            RpcMethod(NoteEditParams, lambda s, u, p: (s.edit_note(u, p.doc_id, p.session_id, p.note_id, p.text), Ack())[1]),
    "note.delete":          RpcMethod(NoteIdParams, lambda s, u, p: (s.delete_note(u, p.doc_id, p.session_id, p.note_id), Ack())[1]),
    "document.get":         RpcMethod(DocIdParams, lambda s, u, p: s.get_document(u, p.doc_id)),
    "document.section_at":  RpcMethod(SectionAtParams, lambda s, u, p: s.section_at(
                                u, p.doc_id, p.page, p.position)),
    "document.context":     RpcMethod(ContextParams, lambda s, u, p: s.get_context(
                                u, p.doc_id, p.section_id)),
    "document.explain":     RpcMethod(ExplainParams, lambda s, u, p: s.explain(u, p.doc_id, p)),
    "document.summarize":   RpcMethod(SummarizeParams, lambda s, u, p: s.summarize(u, p.doc_id, p)),
    "document.ask":         RpcMethod(AskParams, lambda s, u, p: s.ask(u, p.doc_id, p)),
    "document.delete":      RpcMethod(DocIdParams, lambda s, u, p: (s.delete_document(u, p.doc_id), Ack())[1]),
    "document.delete_knowledge": RpcMethod(DocIdParams, lambda s, u, p: (s.delete_knowledge(u, p.doc_id), Ack())[1]),
    "learner.review":       RpcMethod(LimitParams, lambda s, u, p: s.learning.review(u, p.doc_id, p.limit)),
    "learner.history":      RpcMethod(LimitParams, lambda s, u, p: s.learning.history(u, p.doc_id, p.limit)),
    "session.export":       RpcMethod(SessionIdParams, lambda s, u, p: s.export_session(u, p.doc_id, p.session_id)),
    "document.structure":   RpcMethod(DocIdParams, lambda s, u, p: s.learning.structure(u, p.doc_id)),
    "document.knowledge":   RpcMethod(KnowledgeParams, lambda s, u, p: s.learning.knowledge(u, p.doc_id, p.upto)),
    "document.section_knowledge": RpcMethod(SectionParams, lambda s, u, p: s.learning.section_knowledge(u, p.doc_id, p.section_id)),
    "document.growth":      RpcMethod(DocIdParams, lambda s, u, p: s.learning.growth(u, p.doc_id)),
    "concept.prerequisites":   RpcMethod(PrerequisiteParams, lambda s, u, p: s.learning.prerequisites(u, p.doc_id, p.name, p.max_depth)),
    "concept.learning_path":   RpcMethod(LearningPathParams, lambda s, u, p: s.learning.learning_path(u, p.doc_id, p.name)),
    "concept.recommendations": RpcMethod(RecommendParams, lambda s, u, p: s.learning.recommendations(u, p.doc_id, p.name, p.limit)),
    "learner.get":          RpcMethod(DocIdParams, lambda s, u, p: s.learning.learner(u, p.doc_id)),
    "learner.diagnostic_start":  RpcMethod(DiagnosticParams, lambda s, u, p: s.learning.start_diagnostic(u, p.doc_id, p)),
    "learner.diagnostic_finish": RpcMethod(DocIdParams, lambda s, u, p: s.learning.finish_diagnostic(u, p.doc_id)),
    "learner.feedback":     RpcMethod(FeedbackParams, lambda s, u, p: s.learning.feedback(u, p.doc_id, p)),
    "learner.progress":     RpcMethod(ProgressParams, lambda s, u, p: s.learning.progress(u, p.doc_id, p.section_id)),
    "quiz.new":             RpcMethod(QuizParams, lambda s, u, p: s.learning.quiz(u, p.doc_id, p.section_id, p.count)),
    "quiz.answer":          RpcMethod(AnswerParams, lambda s, u, p: s.learning.answer(u, p.doc_id, p.question_id, p.choice)),
}


def _error(id_: Any, code: int, message: str, data: Optional[dict] = None) -> dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "error": err, "id": id_}


def _dispatch(service: DocumentApiService, user: str, req: Any) -> Optional[dict]:
    if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or not isinstance(req.get("method"), str):
        return _error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 request.")
    notification = "id" not in req
    id_ = req.get("id")
    if id_ is not None and (isinstance(id_, bool) or not isinstance(id_, (str, int))):
        return _error(None, INVALID_REQUEST, "id must be a string, a number or null.")

    def reply(result: Any = None, error: Optional[dict] = None) -> Optional[dict]:
        return None if notification else (error or {"jsonrpc": "2.0", "result": result, "id": id_})

    name = req["method"]
    if name == "rpc.discover":
        return reply({n: {"params": m.params.model_json_schema()} for n, m in METHODS.items()})
    method = METHODS.get(name)
    if method is None:
        return reply(error=_error(id_, METHOD_NOT_FOUND, f"Method {name!r} not found."))
    raw = req.get("params", {})
    if not isinstance(raw, dict):
        return reply(error=_error(id_, INVALID_PARAMS, "params must be an object (named parameters)."))
    try:
        params = method.params.model_validate(raw)
    except ValidationError as e:
        details = [{"field": ".".join(map(str, d["loc"])), "message": d["msg"]} for d in e.errors()]
        return reply(error=_error(id_, INVALID_PARAMS, "Invalid params.",
                                  {"code": "invalid_params", "details": details}))
    try:
        result = method.call(service, user, params)
    except ServiceError as e:
        return reply(error=_error(id_, e.rpc_code, e.message, {"code": e.code}))
    except Exception:
        return reply(error=_error(id_, INTERNAL, "The server failed to process the request.",
                                  {"code": "internal_error"}))
    return reply(result.model_dump(mode="json") if isinstance(result, BaseModel) else result)


def handle_payload(service: DocumentApiService, user: str, payload: Any) -> Any:
    """Return the JSON-RPC response (dict / list), or None when nothing is to be sent back."""
    if isinstance(payload, list):
        if not payload:
            return _error(None, INVALID_REQUEST, "Empty batch.")
        if len(payload) > MAX_BATCH:
            return _error(None, INVALID_REQUEST, f"Batch too large (max {MAX_BATCH}).")
        out = [r for r in (_dispatch(service, user, item) for item in payload) if r is not None]
        return out or None
    return _dispatch(service, user, payload)


def build_rpc_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(authorize)])

    @router.post("/rpc", tags=["rpc"], include_in_schema=True)
    async def rpc(request: Request, user: str = Depends(current_user),
                  service: DocumentApiService = Depends(get_service)):
        limit = request.app.state.settings.max_upload_bytes * 4 // 3 + 4096
        body = await request.body()
        if len(body) > limit:
            raise ServiceError("file_too_large", "Request body too large.")
        try:
            payload = json.loads(body)
        except ValueError:
            return _error(None, PARSE_ERROR, "Parse error.")
        result = await run_in_threadpool(handle_payload, service, user, payload)
        return Response(status_code=204) if result is None else result

    return router
