"""REST protocol: resource-style routes under ``/v1``. Thin: parse, call the service, return."""
from __future__ import annotations
import hmac
import json
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from .errors import ServiceError
from .schemas import (
    Ack, ChatBody, ChatResult, ExportOut, HistoryOut, ReviewOut, NoteBody, NoteOut, SessionCreateBody, SessionDetail, SessionList, SessionOut, SessionRenameBody,
    AnswerBody, AnswerOut, AskBody, ConceptResult, ContextOut, DiagnosticBody, DiagnosticOut, DocumentInfo, DocumentList,
    ExplainBody, ExplanationResult, FeedbackBody, GrowthOut, Health, JobInfo, JobList, KnowledgeOut, LearnerOut, ProgressBody,
    QuizBody, QuizOut, SectionKnowledgeOut, SectionRef, StructureOut, SummarizeBody,
)
from .service import DocumentApiService

_USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


def get_service(request: Request) -> DocumentApiService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ServiceError("not_ready", "The server is still starting up.")
    return service


def authorize(request: Request, authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer-token check shared by REST and RPC. No key configured = open (loopback only)."""
    expected = request.app.state.settings.api_key
    if not expected:
        return
    supplied = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise ServiceError("unauthorized", "Missing or invalid API key.")


def current_user(request: Request, x_user_id: Optional[str] = Header(default=None)) -> str:
    user = (x_user_id or request.app.state.settings.default_user_id).strip()
    if not _USER_RE.match(user):
        raise ServiceError("invalid_params", "X-User-Id may only contain letters, digits and _.@-")
    return user


def build_rest_router() -> APIRouter:
    open_router = APIRouter(prefix="/v1")
    router = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])

    @open_router.get("/health", response_model=Health, tags=["system"])
    def health(svc: DocumentApiService = Depends(get_service)):
        return svc.health()

    @router.post("/documents", response_model=JobInfo, status_code=202, tags=["documents"],
                 summary="Upload a document; processing continues in the background")
    def register_document(request: Request, response: Response, file: UploadFile = File(...),
                          build_graph: bool = Form(True),
                          user: str = Depends(current_user),
                          svc: DocumentApiService = Depends(get_service)):
        limit = request.app.state.settings.max_upload_bytes
        data = file.file.read(limit + 1)  # never read more than limit+1 bytes
        job = svc.register_document(user, file.filename or "", data, build_graph)
        response.headers["Location"] = f"/v1/jobs/{job.job_id}"
        return job

    @router.get("/jobs", response_model=JobList, tags=["jobs"])
    def list_jobs(user: str = Depends(current_user),
                  svc: DocumentApiService = Depends(get_service)):
        return svc.list_jobs(user)

    @router.get("/jobs/{job_id}", response_model=JobInfo, tags=["jobs"])
    def get_job(job_id: str, user: str = Depends(current_user),
                svc: DocumentApiService = Depends(get_service)):
        return svc.get_job(user, job_id)

    @router.post("/jobs/{job_id}/pause", response_model=JobInfo, tags=["jobs"], summary="Stop at the next step, keeping the work")
    def pause_job(job_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.pause_job(user, job_id)

    @router.post("/jobs/{job_id}/resume", response_model=JobInfo, tags=["jobs"], summary="Continue a paused job")
    def resume_job(job_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.resume_job(user, job_id)

    @router.post("/jobs/{job_id}/cancel", response_model=JobInfo, tags=["jobs"], summary="Stop and discard the work")
    def cancel_job(job_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.cancel_job(user, job_id)

    @router.get("/documents", response_model=DocumentList, tags=["documents"])
    def list_documents(user: str = Depends(current_user),
                       svc: DocumentApiService = Depends(get_service)):
        return svc.list_documents(user)

    @router.get("/documents/{doc_id}", response_model=DocumentInfo, tags=["documents"])
    def get_document(doc_id: str, user: str = Depends(current_user),
                     svc: DocumentApiService = Depends(get_service)):
        return svc.get_document(user, doc_id)

    @router.get("/documents/{doc_id}/section", response_model=SectionRef, tags=["documents"])
    def section_at(doc_id: str, page: int = Query(ge=0), position: int = Query(ge=0),
                   user: str = Depends(current_user),
                   svc: DocumentApiService = Depends(get_service)):
        return svc.section_at(user, doc_id, page, position)

    @router.get("/documents/{doc_id}/context", response_model=ContextOut, tags=["documents"])
    def context(doc_id: str, section_id: int = Query(default=0, ge=0),
                user: str = Depends(current_user),
                svc: DocumentApiService = Depends(get_service)):
        return svc.get_context(user, doc_id, section_id)

    @router.post("/documents/{doc_id}/explain", response_model=ExplanationResult, tags=["explain"])
    def explain(doc_id: str, body: ExplainBody, user: str = Depends(current_user),
                svc: DocumentApiService = Depends(get_service)):
        return svc.explain(user, doc_id, body)

    @router.post("/documents/{doc_id}/summarize", response_model=ExplanationResult, tags=["explain"])
    def summarize(doc_id: str, body: SummarizeBody, user: str = Depends(current_user),
                  svc: DocumentApiService = Depends(get_service)):
        return svc.summarize(user, doc_id, body)

    @router.post("/documents/{doc_id}/ask", response_model=ExplanationResult, tags=["explain"])
    def ask(doc_id: str, body: AskBody, user: str = Depends(current_user),
            svc: DocumentApiService = Depends(get_service)):
        return svc.ask(user, doc_id, body)

    # ---- the conversation: streamed replies, sessions, notes
    @router.post("/documents/{doc_id}/chat/stream", tags=["chat"],
                 summary="One turn of the conversation as a server-sent event stream: session, status, delta..., done | error")
    def chat_stream(doc_id: str, body: ChatBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        session, events = svc.chat_stream(user, doc_id, body)      # validation errors surface as normal JSON errors

        def sse():
            yield f"event: session\ndata: {json.dumps(session.model_dump(mode='json'))}\n\n"
            for ev in events:
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/documents/{doc_id}/chat", response_model=ChatResult, tags=["chat"], summary="The same turn, answered in one piece")
    def chat(doc_id: str, body: ChatBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.chat_once(user, doc_id, body)

    @router.get("/documents/{doc_id}/sessions", response_model=SessionList, tags=["chat"])
    def list_sessions(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.list_sessions(user, doc_id)

    @router.post("/documents/{doc_id}/sessions", response_model=SessionOut, status_code=201, tags=["chat"])
    def create_session(doc_id: str, body: SessionCreateBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.create_session(user, doc_id, body.title)

    @router.get("/documents/{doc_id}/sessions/{session_id}", response_model=SessionDetail, tags=["chat"])
    def get_session(doc_id: str, session_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.get_session(user, doc_id, session_id)

    @router.patch("/documents/{doc_id}/sessions/{session_id}", response_model=SessionOut, tags=["chat"])
    def rename_session(doc_id: str, session_id: str, body: SessionRenameBody, user: str = Depends(current_user),
                       svc: DocumentApiService = Depends(get_service)):
        return svc.rename_session(user, doc_id, session_id, body.title)

    @router.delete("/documents/{doc_id}/sessions/{session_id}", status_code=204, tags=["chat"])
    def delete_session(doc_id: str, session_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        svc.delete_session(user, doc_id, session_id)
        return Response(status_code=204)

    @router.post("/documents/{doc_id}/sessions/{session_id}/notes", response_model=NoteOut, status_code=201, tags=["chat"])
    def add_note(doc_id: str, session_id: str, body: NoteBody, user: str = Depends(current_user),
                 svc: DocumentApiService = Depends(get_service)):
        return svc.add_note(user, doc_id, session_id, body.text, body.kind)

    @router.put("/documents/{doc_id}/sessions/{session_id}/notes/{note_id}", status_code=204, tags=["chat"])
    def edit_note(doc_id: str, session_id: str, note_id: str, body: NoteBody,
                  user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        svc.edit_note(user, doc_id, session_id, note_id, body.text)
        return Response(status_code=204)

    @router.delete("/documents/{doc_id}/sessions/{session_id}/notes/{note_id}", status_code=204, tags=["chat"])
    def delete_note(doc_id: str, session_id: str, note_id: str, user: str = Depends(current_user),
                    svc: DocumentApiService = Depends(get_service)):
        svc.delete_note(user, doc_id, session_id, note_id)
        return Response(status_code=204)

    @router.post("/documents/{doc_id}/sessions/{session_id}/summary", response_model=NoteOut, status_code=201, tags=["chat"],
                 summary="Have the tutor write a summary of the conversation, kept as a note")
    def summarize_session(doc_id: str, session_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.summarize_session(user, doc_id, session_id)

    # ---- the document's structure and its knowledge graph
    @router.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
    def delete_document(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        svc.delete_document(user, doc_id)
        return Response(status_code=204)

    @router.delete("/documents/{doc_id}/knowledge", status_code=204, tags=["knowledge"],
                   summary="Drop the knowledge graph and this reader's tracking; the document stays")
    def delete_knowledge(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        svc.delete_knowledge(user, doc_id)
        return Response(status_code=204)

    @router.get("/documents/{doc_id}/review", response_model=ReviewOut, tags=["learner"],
                summary="Concepts the reader is likely forgetting, weakest first")
    def review(doc_id: str, limit: int = Query(default=20, ge=1, le=200), user: str = Depends(current_user),
               svc: DocumentApiService = Depends(get_service)):
        return svc.learning.review(user, doc_id, limit)

    @router.get("/documents/{doc_id}/history", response_model=HistoryOut, tags=["learner"],
                summary="How belief in each concept changed over time")
    def history(doc_id: str, limit: int = Query(default=200, ge=1, le=1000), user: str = Depends(current_user),
                svc: DocumentApiService = Depends(get_service)):
        return svc.learning.history(user, doc_id, limit)

    @router.get("/documents/{doc_id}/sessions/{session_id}/export", response_model=ExportOut, tags=["chat"],
                summary="A conversation and its notes as Markdown")
    def export_session(doc_id: str, session_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.export_session(user, doc_id, session_id)

    @router.get("/documents/{doc_id}/structure", response_model=StructureOut, tags=["knowledge"])
    def structure(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.structure(user, doc_id)

    @router.get("/documents/{doc_id}/knowledge", response_model=KnowledgeOut, tags=["knowledge"],
                summary="The knowledge graph as of a reading position (whole document if omitted)")
    def knowledge(doc_id: str, upto: Optional[int] = Query(default=None, ge=0), user: str = Depends(current_user),
                  svc: DocumentApiService = Depends(get_service)):
        return svc.learning.knowledge(user, doc_id, upto)

    @router.get("/documents/{doc_id}/sections/{section_id}/knowledge", response_model=SectionKnowledgeOut, tags=["knowledge"])
    def section_knowledge(doc_id: str, section_id: int, user: str = Depends(current_user),
                          svc: DocumentApiService = Depends(get_service)):
        return svc.learning.section_knowledge(user, doc_id, section_id)

    @router.get("/documents/{doc_id}/growth", response_model=GrowthOut, tags=["knowledge"])
    def growth(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.growth(user, doc_id)

    @router.get("/documents/{doc_id}/concepts/{name}/prerequisites", response_model=ConceptResult, tags=["concepts"])
    def prerequisites(doc_id: str, name: str, max_depth: int = Query(default=3, ge=1, le=10), user: str = Depends(current_user),
                      svc: DocumentApiService = Depends(get_service)):
        return svc.learning.prerequisites(user, doc_id, name, max_depth)

    @router.get("/documents/{doc_id}/concepts/{name}/learning-path", response_model=ConceptResult, tags=["concepts"])
    def learning_path(doc_id: str, name: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.learning_path(user, doc_id, name)

    @router.get("/documents/{doc_id}/concepts/{name}/recommendations", response_model=ConceptResult, tags=["concepts"])
    def recommendations(doc_id: str, name: str, limit: int = Query(default=5, ge=1, le=50), user: str = Depends(current_user),
                        svc: DocumentApiService = Depends(get_service)):
        return svc.learning.recommendations(user, doc_id, name, limit)

    # ---- the learner
    @router.get("/documents/{doc_id}/learner", response_model=LearnerOut, tags=["learner"])
    def learner(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.learner(user, doc_id)

    @router.post("/documents/{doc_id}/diagnostic", response_model=DiagnosticOut, tags=["learner"],
                 summary="Say where you stand and get a few questions that pin it down")
    def start_diagnostic(doc_id: str, body: DiagnosticBody, user: str = Depends(current_user),
                         svc: DocumentApiService = Depends(get_service)):
        return svc.learning.start_diagnostic(user, doc_id, body)

    @router.post("/documents/{doc_id}/diagnostic/finish", response_model=DiagnosticOut, tags=["learner"])
    def finish_diagnostic(doc_id: str, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.finish_diagnostic(user, doc_id)

    @router.post("/documents/{doc_id}/quiz", response_model=QuizOut, tags=["learner"])
    def quiz(doc_id: str, body: QuizBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.quiz(user, doc_id, body.section_id, body.count)

    @router.post("/documents/{doc_id}/quiz/{question_id}/answer", response_model=AnswerOut, tags=["learner"])
    def answer(doc_id: str, question_id: str, body: AnswerBody, user: str = Depends(current_user),
               svc: DocumentApiService = Depends(get_service)):
        return svc.learning.answer(user, doc_id, question_id, body.choice)

    @router.post("/documents/{doc_id}/feedback", response_model=Ack, tags=["learner"])
    def feedback(doc_id: str, body: FeedbackBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.feedback(user, doc_id, body)

    @router.post("/documents/{doc_id}/progress", response_model=Ack, tags=["learner"],
                 summary="The reader finished a section")
    def progress(doc_id: str, body: ProgressBody, user: str = Depends(current_user), svc: DocumentApiService = Depends(get_service)):
        return svc.learning.progress(user, doc_id, body.section_id)

    combined = APIRouter()
    combined.include_router(open_router)
    combined.include_router(router)
    return combined
