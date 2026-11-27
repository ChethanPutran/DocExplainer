"""DocumentApiService: the one implementation behind both REST and JSON-RPC.

It owns what a service must own and the core must not: upload validation and storage, which user owns
which document, background jobs, error translation, and serialisation of results. Everything about
documents, knowledge, learners and tutoring is delegated to the orchestrator's ``System``.

Registering a document is slow (a model call per section), so it runs as a background job on a single worker
thread. Nothing else waits for it: every store is thread-safe, and a document being ingested is not yet visible
to questions, so reading, asking and quizzing carry on normally. Scale out with more server processes.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel

from doc_explainer.core.chat import ChatError
from doc_explainer.core.common.ports import LLMError
from doc_explainer.core.document import ParseError, UnsupportedFormat
from doc_explainer.orchestrator import EmptyDocumentError

from .errors import ServiceError
from .jobs import CANCELLED, FAILED, PAUSED, QUEUED, RUNNING, SUCCEEDED, JobStore, now
from .learning import LearningApi
from .schemas import (AskBody, ChatBody, ChatResult, ContextOut, DocumentInfo, DocumentList, ExplainBody, ExplanationOut, ExportOut,
                      ExplanationResult, Health, JobError, JobInfo, JobList, MessageOut, NoteOut, SectionRef, SessionDetail,
                      SessionList, SessionOut, SummarizeBody)
from .settings import ApiSettings

logger = logging.getLogger(__name__)

API_VERSION = "v1"

# What the reader sees while a reply is being prepared (the stages come from TutorService.stream).
STAGE_TEXT = {"understanding": "Understanding your request", "searching": "Looking through the document",
              "thinking": "Thinking it through", "writing": "Writing the reply", "finishing": "Finishing up"}


class JobStopped(Exception):
    """Raised inside a running job (from its progress callback) when the user paused or cancelled it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason       # "pause" | "cancel"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("doc-explainer")
    except Exception:
        return "unknown"


def _public(record: dict) -> dict:
    """The fields of a registry record that clients see (no paths, hashes or owner)."""
    return {k: record.get(k, d) for k, d in (("doc_id", ""), ("title", ""), ("filename", ""), ("size_bytes", 0), ("uploaded_at", ""),
                                             ("knowledge", "none"), ("sections", 0), ("concepts", 0))}


def _dump(model: BaseModel) -> dict:
    try:
        return model.model_dump(mode="json", fallback=str)
    except TypeError:  # older pydantic without ``fallback``
        return json.loads(model.model_dump_json())


class _Registry:
    """Which documents exist and who owns them. Persisted so listings survive a restart."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._docs: Dict[str, dict] = {}
        if path.exists():
            try:
                self._docs = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                bad = path.with_suffix(".corrupt")
                logger.error("Document registry %s is unreadable; moved to %s", path, bad)
                path.replace(bad)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, indent=2)
        os.replace(tmp, self._path)

    def add(self, doc_id: str, record: dict) -> None:
        with self._lock:
            self._docs[doc_id] = record
            self._save()

    def get(self, doc_id: str, user_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._docs.get(doc_id)
        return rec if rec and rec["user_id"] == user_id else None

    def find_by_hash(self, user_id: str, sha256: str, build_graph: bool) -> Optional[dict]:
        """Already-registered identical content (with a knowledge graph if the caller wants one)."""
        with self._lock:
            for r in self._docs.values():
                if (r["user_id"] == user_id and r.get("sha256") == sha256
                        and (r.get("build_graph", True) or not build_graph)):
                    return dict(r)
        return None

    def forget(self, doc_id: str) -> None:
        with self._lock:
            if self._docs.pop(doc_id, None) is not None:
                self._save()

    def list(self, user_id: str) -> List[dict]:
        with self._lock:
            recs = [r for r in self._docs.values() if r["user_id"] == user_id]
        return sorted(recs, key=lambda r: r["uploaded_at"], reverse=True)


class DocumentApiService:
    def __init__(self, system: Any, settings: ApiSettings):
        self._sys = system
        self._settings = settings
        self._registry = _Registry(settings.data_dir / "documents.json")
        self._jobs = JobStore(settings.data_dir / "jobs.json")
        self._uploads = settings.data_dir / "uploads"
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._closing = False
        self._control: Dict[str, str] = {}     # job id -> "pause" | "cancel", read by the running job between steps
        self.learning = LearningApi(system, self._owned)

    def close(self) -> None:
        """Stop accepting work and ask a running ingest to stop between steps."""
        self._closing = True
        self._worker.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ helpers
    def _owned(self, user_id: str, doc_id: str) -> dict:
        rec = self._registry.get(doc_id, user_id)
        if rec is None:  # unknown id and someone else's id look the same on purpose
            raise ServiceError("document_not_found", f"Document {doc_id!r} not found.")
        return rec

    @staticmethod
    def _translate(e: BaseException) -> ServiceError:
        if isinstance(e, ServiceError):
            return e
        if isinstance(e, ChatError):
            return ServiceError(e.code, str(e))
        if isinstance(e, UnsupportedFormat):
            return ServiceError("unsupported_file_type", str(e))
        if isinstance(e, (ParseError, EmptyDocumentError)):
            return ServiceError("processing_failed", str(e))
        if isinstance(e, LLMError):
            return ServiceError("processing_failed", f"The language model call failed: {e}")
        if isinstance(e, ValueError):
            return ServiceError("processing_failed", str(e))
        logger.error("Unexpected failure", exc_info=e)
        return ServiceError("internal_error", "The server failed to process the request.")

    def _guard(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - translated into a stable error
            raise self._translate(e) from e

    # ------------------------------------------------------------------ system
    def supported_suffixes(self) -> List[str]:
        return self._sys.supported_extensions()

    def health(self) -> Health:
        return Health(status="ok", version=_version(), api_version=API_VERSION, protocols=self._settings.protocols,
                      supported_types=self.supported_suffixes(), jobs_pending=self._jobs.pending())

    # ------------------------------------------------------------------ documents (background jobs)
    def register_document(self, user_id: str, filename: str, data: bytes, build_graph: bool = True) -> JobInfo:
        """Validate and store the upload, then process it in the background. Returns the job at once.

        Cheap problems (bad name/type/size) fail here, synchronously. Identical content that is already
        registered, or already being processed, is not processed twice. ``build_graph`` = build the knowledge graph."""
        name = Path(filename.replace("\\", "/")).name.strip()
        if not name or name.startswith("."):
            raise ServiceError("invalid_params", "A file name is required.")
        allowed = self.supported_suffixes()
        if Path(name).suffix.lower() not in allowed:
            raise ServiceError("unsupported_file_type", "Supported file types: " + ", ".join(allowed))
        if not data:
            raise ServiceError("invalid_params", "The file is empty.")
        if len(data) > self._settings.max_upload_bytes:
            raise ServiceError("file_too_large", f"Maximum upload size is {self._settings.max_upload_mb} MB.")

        sha = hashlib.sha256(data).hexdigest()
        existing = self._registry.find_by_hash(user_id, sha, build_graph)
        if existing is None and build_graph:
            # Registered earlier WITHOUT a graph and now wanted with one: build it on the SAME document, not a copy.
            earlier = self._registry.find_by_hash(user_id, sha, False)
            if earlier and earlier.get("knowledge", "none") == "none":
                existing = earlier
        # A document whose knowledge graph failed, is partial or was skipped is not "done" when a graph is wanted:
        # uploading it again resumes the SAME document (finished sections are kept, failed ones are retried).
        resume = bool(existing and build_graph and existing.get("knowledge", "none") in ("failed", "partial", "none"))
        if existing and not resume:
            job = self._jobs.create(user_id, name, state=SUCCEEDED, started_at=now(), finished_at=now(),
                                    document=_public(existing), reused=True, progress=1.0, stage="done")
            return self._job_info(job)
        active = self._jobs.find_active(user_id, sha, build_graph)
        if active:
            return self._job_info(active)
        if (self._jobs.pending(user_id) >= self._settings.max_jobs_per_user
                or self._jobs.pending() >= self._settings.max_queued_jobs):
            raise ServiceError("queue_full", "Too many documents are waiting to be processed. Wait for one to finish and try again.")

        folder = self._uploads / uuid.uuid4().hex
        folder.mkdir(parents=True)
        path = folder / name
        path.write_bytes(data)
        job = self._jobs.create(user_id, name, sha256=sha, build_graph=build_graph, path=str(path), upload_dir=str(folder),
                                size_bytes=len(data), doc_id=existing["doc_id"] if resume else uuid.uuid4().hex, resume=resume)
        try:
            self._worker.submit(self._run_job, job["job_id"])
        except RuntimeError:  # shutting down
            shutil.rmtree(folder, ignore_errors=True)
            self._jobs.update(job["job_id"], state=FAILED, finished_at=now(), error={"code": "not_ready", "message": "The server is shutting down."})
            raise ServiceError("not_ready", "The server is shutting down.")
        return self._job_info(job)

    def register_document_b64(self, user_id: str, filename: str, content_b64: str, build_graph: bool = True) -> JobInfo:
        if len(content_b64) > self._settings.max_upload_bytes * 4 // 3 + 16:
            raise ServiceError("file_too_large", f"Maximum upload size is {self._settings.max_upload_mb} MB.")
        try:
            data = base64.b64decode(content_b64, validate=True)
        except (binascii.Error, ValueError):
            raise ServiceError("invalid_params", "content_b64 is not valid base64.")
        return self.register_document(user_id, filename, data, build_graph)

    def _progress_callback(self, job_id: str) -> Callable[[str, float], None]:
        last = {"t": 0.0, "f": -1.0, "s": ""}

        def report(stage: str, fraction: float) -> None:
            reason = self._control.get(job_id)
            if reason:
                raise JobStopped(reason)           # the user paused or cancelled: stop at this step boundary
            t = time.monotonic()
            if stage != last["s"] or fraction - last["f"] >= 0.01 or t - last["t"] >= 2.0:
                last.update(t=t, f=fraction, s=stage)
                self._jobs.update(job_id, progress=round(min(max(fraction, 0.0), 1.0), 3), stage=stage)
        return report

    def _run_job(self, job_id: str) -> None:
        """Worker thread: ingest one document. Never raises; the outcome is recorded on the job."""
        current = self._jobs.peek(job_id)
        if current is None or current["state"] != QUEUED:       # paused or cancelled while it waited its turn
            return
        job = self._jobs.update(job_id, state=RUNNING, started_at=now(), stage="starting")
        doc_id = job["doc_id"]
        try:
            report = self._sys.ingest(job["path"], doc_id, build_knowledge=job["build_graph"],
                                      progress=self._progress_callback(job_id), cancelled=lambda: self._closing)
            record = {
                "doc_id": doc_id, "user_id": job["user_id"], "title": report.title or job["filename"],
                "filename": job["filename"], "size_bytes": job["size_bytes"], "path": job["path"],
                "uploaded_at": job["created_at"], "sha256": job["sha256"], "build_graph": job["build_graph"],
                "knowledge": report.knowledge, "knowledge_error": report.knowledge_error,
                "sections": report.sections, "concepts": report.concepts,
            }
            self._registry.add(doc_id, record)
            self._jobs.update(job_id, state=SUCCEEDED, finished_at=now(), document=_public(record), progress=1.0, stage="done")
        except JobStopped as stop:
            self._stopped(job, stop.reason)
        except Exception as e:  # noqa: BLE001
            err = self._translate(e)
            self._fail_job(job, err.code, err.message)
        finally:
            self._control.pop(job_id, None)

    def _stopped(self, job: dict, reason: str) -> None:
        if reason == "pause":       # keep everything: resuming continues from the sections already done
            self._jobs.update(job["job_id"], state=PAUSED, stage="paused", started_at=None)
        else:
            self._discard(job)
            self._jobs.update(job["job_id"], state=CANCELLED, finished_at=now(), stage="cancelled")

    def _discard(self, job: dict) -> None:
        """Throw away an unfinished job's upload and whatever it stored, but never a document that is already registered."""
        shutil.rmtree(job.get("upload_dir") or "", ignore_errors=True)
        if self._registry.get(job.get("doc_id", ""), job["user_id"]) is None:
            try:
                self._sys.remove(job.get("doc_id", ""))
            except Exception:
                logger.exception("Could not clean up after job %s", job["job_id"])

    def _fail_job(self, job: dict, code: str, message: str) -> None:
        self._discard(job)
        self._jobs.update(job["job_id"], state=FAILED, finished_at=now(), error={"code": code, "message": message})

    # ------------------------------------------------------------------ job control
    def _own_job(self, user_id: str, job_id: str) -> dict:
        job = self._jobs.get(job_id, user_id)
        if job is None:
            raise ServiceError("job_not_found", f"Job {job_id!r} not found.")
        return job

    def pause_job(self, user_id: str, job_id: str) -> JobInfo:
        """Stop at the next step and keep everything. A queued job just waits; resume continues it."""
        job = self._own_job(user_id, job_id)
        if job["state"] == QUEUED:
            return self._job_info(self._jobs.update(job_id, state=PAUSED, stage="paused"))
        if job["state"] == RUNNING:
            self._control[job_id] = "pause"
            return self._job_info(self._jobs.update(job_id, stage="pausing after the current step"))
        if job["state"] == PAUSED:
            return self._job_info(job)
        raise ServiceError("job_not_active", f"This job is already {job['state']}.")

    def resume_job(self, user_id: str, job_id: str) -> JobInfo:
        job = self._own_job(user_id, job_id)
        if job["state"] == PAUSED:
            self._control.pop(job_id, None)
            job = self._jobs.update(job_id, state=QUEUED, stage="queued")
            try:
                self._worker.submit(self._run_job, job_id)
            except RuntimeError:
                raise ServiceError("not_ready", "The server is shutting down.")
            return self._job_info(job)
        if job["state"] == RUNNING and self._control.get(job_id) == "pause":     # changed their mind before it stopped
            self._control.pop(job_id, None)
            return self._job_info(self._jobs.update(job_id, stage="resuming"))
        if job["state"] in (QUEUED, RUNNING):
            return self._job_info(job)
        raise ServiceError("job_not_active", f"This job is already {job['state']}.")

    def cancel_job(self, user_id: str, job_id: str) -> JobInfo:
        """Stop and throw the work away (an already registered document is left alone)."""
        job = self._own_job(user_id, job_id)
        if job["state"] in (QUEUED, PAUSED):
            self._discard(job)
            return self._job_info(self._jobs.update(job_id, state=CANCELLED, finished_at=now(), stage="cancelled"))
        if job["state"] == RUNNING:
            self._control[job_id] = "cancel"
            return self._job_info(self._jobs.update(job_id, stage="cancelling after the current step"))
        if job["state"] == CANCELLED:
            return self._job_info(job)
        raise ServiceError("job_not_active", f"This job is already {job['state']}.")

    def _job_info(self, job: dict) -> JobInfo:
        return JobInfo(
            job_id=job["job_id"], kind=job["kind"], state=job["state"], filename=job["filename"],
            created_at=job["created_at"], started_at=job["started_at"], finished_at=job["finished_at"],
            queue_position=self._jobs.queue_position(job["job_id"]), reused=job.get("reused", False),
            progress=job.get("progress", 0.0), stage=job.get("stage", ""),
            document=DocumentInfo(**job["document"]) if job["document"] else None,
            error=JobError(**job["error"]) if job["error"] else None)

    def get_job(self, user_id: str, job_id: str) -> JobInfo:
        job = self._jobs.get(job_id, user_id)
        if job is None:  # unknown and someone else's look the same on purpose
            raise ServiceError("job_not_found", f"Job {job_id!r} not found.")
        return self._job_info(job)

    def list_jobs(self, user_id: str) -> JobList:
        return JobList(jobs=[self._job_info(j) for j in self._jobs.list(user_id)])

    def list_documents(self, user_id: str) -> DocumentList:
        return DocumentList(documents=[DocumentInfo(**_public(r)) for r in self._registry.list(user_id)])

    def get_document(self, user_id: str, doc_id: str) -> DocumentInfo:
        return DocumentInfo(**_public(self._owned(user_id, doc_id)))

    def delete_document(self, user_id: str, doc_id: str) -> None:
        rec = self._owned(user_id, doc_id)
        self._guard(lambda: self._sys.remove(doc_id))
        self._registry.forget(doc_id)
        self._delete_upload(rec)

    def _delete_upload(self, rec: dict) -> None:
        """Remove the stored copy of the original file (its own upload folder, never anything outside the uploads root)."""
        path = rec.get("path")
        if not path:
            return
        folder = Path(path).resolve().parent
        if folder.parent == self._uploads.resolve():
            shutil.rmtree(folder, ignore_errors=True)

    def delete_knowledge(self, user_id: str, doc_id: str) -> None:
        """Drop the document's knowledge graph and this reader's tracking for it. The document stays readable and searchable."""
        rec = self._owned(user_id, doc_id)
        self._guard(lambda: self._sys.knowledge_store.reset(doc_id))
        self._guard(lambda: self._sys.tracker.forget(user_id, doc_id))
        self._registry.add(doc_id, {**rec, "knowledge": "none", "knowledge_error": "", "concepts": 0, "build_graph": False})

    def export_session(self, user_id: str, doc_id: str, session_id: str) -> ExportOut:
        """The conversation and its notes as one Markdown document."""
        import datetime as dt
        rec = self._owned(user_id, doc_id)
        detail = self.get_session(user_id, doc_id, session_id)

        def when(ts: float) -> str:
            return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        title = detail.session.title or "Conversation"
        lines = [f"# {title}", "", f"*Document: {rec.get('title') or rec.get('filename', '')}*", ""]
        for m in detail.messages:
            lines += [f"**{'You' if m.role == 'user' else 'Tutor'}** ({when(m.created_at)})", "", m.text.strip(), ""]
        if detail.notes:
            lines += ["---", "", "## Notes", ""]
            for n in detail.notes:
                lines += [f"### {'Summary' if n.kind == 'summary' else 'Note'} ({when(n.created_at)})", "", n.text.strip(), ""]
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "conversation"
        return ExportOut(filename=f"{safe}.md", markdown="\n".join(lines))

    # ------------------------------------------------------------------ position, context
    def section_at(self, user_id: str, doc_id: str, page: int, position: int) -> SectionRef:
        self._owned(user_id, doc_id)
        return SectionRef(section_id=int(self._guard(lambda: self._sys.section_at(doc_id, page))))

    def get_context(self, user_id: str, doc_id: str, section_id: int = 0) -> ContextOut:
        self._owned(user_id, doc_id)
        return self._guard(lambda: self.learning.context(user_id, doc_id, section_id))

    # ------------------------------------------------------------------ tutoring
    @staticmethod
    def _result(answer) -> ExplanationResult:
        return ExplanationResult(text=answer.text, explanation=ExplanationOut.model_validate(_dump(answer.explanation)))

    def explain(self, user_id: str, doc_id: str, body: ExplainBody) -> ExplanationResult:
        self._owned(user_id, doc_id)
        level = body.style.level.value if body.style else None
        if body.session_id:
            return self.chat_once(user_id, doc_id, ChatBody(kind="explain", text=body.selected_text, section_id=body.section_id,
                                                            session_id=body.session_id, level=level))
        return self._guard(lambda: self._result(self._sys.tutor.explain(user_id, doc_id, body.selected_text, body.section_id, level)))

    def summarize(self, user_id: str, doc_id: str, body: SummarizeBody) -> ExplanationResult:
        self._owned(user_id, doc_id)
        if body.session_id:
            return self.chat_once(user_id, doc_id, ChatBody(kind="summarize", text=body.selected_text, section_id=body.section_id,
                                                            session_id=body.session_id))
        return self._guard(lambda: self._result(self._sys.tutor.summarize(user_id, doc_id, body.selected_text, body.section_id)))

    def ask(self, user_id: str, doc_id: str, body: AskBody) -> ExplanationResult:
        self._owned(user_id, doc_id)
        if body.session_id:
            return self.chat_once(user_id, doc_id, ChatBody(kind="ask", text=body.question, section_id=body.section_id,
                                                            session_id=body.session_id))
        return self._guard(lambda: self._result(self._sys.tutor.ask(user_id, doc_id, body.question, body.section_id)))

    # ------------------------------------------------------------------ the conversation (streamed)
    def chat_stream(self, user_id: str, doc_id: str, body: ChatBody) -> Tuple[SessionOut, Iterator[dict]]:
        """Validate and record the question NOW (so a bad document/session is a normal error), then return the events:
        {"event": "status"|"delta"|"done"|"error", "data": {...}}. The answer is recorded when it is complete."""
        self._owned(user_id, doc_id)
        chat = self._sys.chat
        session = self._guard(lambda: chat.open_session(user_id, doc_id, body.session_id))
        self._guard(lambda: chat.record_question(session, body.text, body.kind, {"section_id": body.section_id}))
        level = body.level.value if body.level else None

        def events() -> Iterator[dict]:
            try:
                for kind, payload in self._sys.tutor.stream(body.kind, user_id, doc_id, body.text, body.section_id, level):
                    if kind == "status":
                        yield {"event": "status", "data": {"stage": payload, "text": STAGE_TEXT.get(str(payload), str(payload))}}
                    elif kind == "delta":
                        yield {"event": "delta", "data": {"text": payload}}
                    else:
                        answer = payload
                        exp = ExplanationOut.model_validate(_dump(answer.explanation))
                        message = chat.record_answer(session, answer.text, body.kind, {
                            "follow_ups": exp.follow_up_questions, "section_id": body.section_id, "concept_ids": answer.concept_ids,
                            "resources": [r.model_dump(mode="json") for r in exp.resources]})
                        result = ChatResult(text=answer.text, explanation=exp, processing_time_ms=exp.metadata.processing_time_ms if exp.metadata else 0.0,
                                            session_id=session.id, message_id=message.id)
                        yield {"event": "done", "data": result.model_dump(mode="json")}
            except Exception as e:  # noqa: BLE001 - told to the client in-band: the HTTP status is already sent
                err = self._translate(e)
                yield {"event": "error", "data": {"code": err.code, "message": err.message}}
        return self._session_out(session), events()

    def chat_once(self, user_id: str, doc_id: str, body: ChatBody) -> ChatResult:
        """The same turn without streaming (JSON-RPC and plain clients)."""
        _, events = self.chat_stream(user_id, doc_id, body)
        for ev in events:
            if ev["event"] == "done":
                return ChatResult.model_validate(ev["data"])
            if ev["event"] == "error":
                raise ServiceError(ev["data"]["code"], ev["data"]["message"])
        raise ServiceError("internal_error", "The reply ended without an answer.")

    # ------------------------------------------------------------------ sessions, notes, summaries
    @staticmethod
    def _session_out(s) -> SessionOut:
        return SessionOut(id=s.id, doc_id=s.doc_id, title=s.title, created_at=s.created_at, updated_at=s.updated_at,
                          messages=s.messages, notes=s.notes)

    def list_sessions(self, user_id: str, doc_id: str) -> SessionList:
        self._owned(user_id, doc_id)
        return SessionList(sessions=[self._session_out(s) for s in self._sys.chat.list_sessions(user_id, doc_id)])

    def create_session(self, user_id: str, doc_id: str, title: str = "") -> SessionOut:
        self._owned(user_id, doc_id)
        return self._session_out(self._guard(lambda: self._sys.chat.create_session(user_id, doc_id, title)))

    def get_session(self, user_id: str, doc_id: str, session_id: str) -> SessionDetail:
        self._owned(user_id, doc_id)
        session, messages, notes = self._guard(lambda: self._sys.chat.history(user_id, doc_id, session_id))
        return SessionDetail(
            session=self._session_out(session),
            messages=[MessageOut(id=m.id, role=m.role, text=m.text, kind=m.kind, created_at=m.created_at, meta=m.meta) for m in messages],
            notes=[NoteOut(id=n.id, kind=n.kind, text=n.text, created_at=n.created_at) for n in notes])

    def rename_session(self, user_id: str, doc_id: str, session_id: str, title: str) -> SessionOut:
        self._owned(user_id, doc_id)
        return self._session_out(self._guard(lambda: self._sys.chat.rename(user_id, doc_id, session_id, title)))

    def delete_session(self, user_id: str, doc_id: str, session_id: str) -> None:
        self._owned(user_id, doc_id)
        self._guard(lambda: self._sys.chat.delete(user_id, doc_id, session_id))

    def add_note(self, user_id: str, doc_id: str, session_id: str, text: str, kind: str = "note") -> NoteOut:
        self._owned(user_id, doc_id)
        n = self._guard(lambda: self._sys.chat.add_note(user_id, doc_id, session_id, text, kind))
        return NoteOut(id=n.id, kind=n.kind, text=n.text, created_at=n.created_at)

    def edit_note(self, user_id: str, doc_id: str, session_id: str, note_id: str, text: str) -> None:
        self._owned(user_id, doc_id)
        self._guard(lambda: self._sys.chat.edit_note(user_id, doc_id, session_id, note_id, text))

    def delete_note(self, user_id: str, doc_id: str, session_id: str, note_id: str) -> None:
        self._owned(user_id, doc_id)
        self._guard(lambda: self._sys.chat.delete_note(user_id, doc_id, session_id, note_id))

    def summarize_session(self, user_id: str, doc_id: str, session_id: str) -> NoteOut:
        """A model-written summary of the conversation, kept as a note of the session."""
        self._owned(user_id, doc_id)
        n = self._guard(lambda: self._sys.chat.summarize(user_id, doc_id, session_id))
        return NoteOut(id=n.id, kind=n.kind, text=n.text, created_at=n.created_at)
