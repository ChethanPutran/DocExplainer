"""``DocExplainerClient`` (sync) and ``AsyncDocExplainerClient``: the one way to use the service.

    client = DocExplainerClient("http://localhost:8000", protocol="rest")   # or protocol="rpc"
    doc = client.register_document("paper.pdf")
    print(client.ask(doc.doc_id, "What is the main finding?").text)

The protocol only changes how requests travel; every method returns the same typed result either
way. Nothing is retried automatically: an explain call can take a while and is not free.

Registering a document is slow (the server calls a model for every section), so it is a background
job on the server. ``register_document`` waits for it and returns the ``Document``; use
``start_registration`` + ``get_job`` / ``wait_for_job`` to stay in control (show progress, or resume
waiting after your own program restarts).
"""
from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path
import json
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Union

import httpx

from . import models as m
from .errors import (DocExplainerError, JobCancelledError, JobTimeoutError, ProtocolError, ServerUnavailableError, error_from_code)
from .protocols import OPERATIONS, PROTOCOLS, RequestSpec, make_protocol

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120.0  # model calls are slow

Style = Union[m.ExplanationStyle, Dict[str, Any]]


class _Base:
    def __init__(self, base_url: Optional[str], protocol: Optional[str], api_key: Optional[str],
                 user_id: Optional[str], timeout: float):
        self.base_url = (base_url or os.getenv("DOC_EXPLAINER_URL") or DEFAULT_URL).rstrip("/")
        self.protocol = (protocol or os.getenv("DOC_EXPLAINER_PROTOCOL") or "rest").lower()
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        self._proto = make_protocol(self.protocol, self.base_url)
        self._timeout = timeout
        headers = {"User-Agent": "doc-explainer-sdk/1.0", "Accept": "application/json"}
        key = api_key if api_key is not None else os.getenv("DOC_EXPLAINER_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if user_id:
            headers["X-User-Id"] = user_id
        self._headers = headers

    def __repr__(self) -> str:  # never includes the API key
        return f"{type(self).__name__}(base_url={self.base_url!r}, protocol={self.protocol!r})"

    # ---- request preparation / error translation shared by sync and async
    def _prepare(self, op_name: str, params: Dict[str, Any]) -> RequestSpec:
        return self._proto.build(OPERATIONS[op_name], params)

    def _kwargs(self, spec: RequestSpec) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"headers": self._headers, "timeout": self._timeout}
        for name in ("params", "json", "data", "files"):
            value = getattr(spec, name)
            if value is not None:
                kw[name] = value
        return kw

    def _unavailable(self, exc: httpx.HTTPError) -> ServerUnavailableError:
        if isinstance(exc, httpx.TimeoutException):
            msg = f"The server at {self.base_url} did not answer within {self._timeout:g}s."
        else:
            msg = f"Cannot reach the Doc Explainer server at {self.base_url}. Is it running?"
        return ServerUnavailableError(msg, code="unavailable")

    @staticmethod
    def _style(style: Optional[Style]) -> Optional[dict]:
        if style is None:
            return None
        if isinstance(style, m.ExplanationStyle):
            return style.model_dump(mode="json", exclude_none=True)
        return {k: (v.value if hasattr(v, "value") else v) for k, v in style.items()}

    @staticmethod
    def _upload(source: Union[str, Path, bytes], filename: Optional[str]):
        if isinstance(source, (bytes, bytearray)):
            if not filename:
                raise ValueError("filename is required when uploading bytes")
            return filename, bytes(source)
        path = Path(source)
        return filename or path.name, path.read_bytes()

    @staticmethod
    def _section(v: Any) -> int:
        return int(v["section_id"])

    @staticmethod
    def _failed(job: "m.Job") -> DocExplainerError:
        e = job.error
        return error_from_code(e.code if e else "processing_failed",
                               e.message if e else f"Job {job.job_id} failed.")

    @staticmethod
    def _timed_out(job_id: str, job: Optional["m.Job"], timeout: float) -> JobTimeoutError:
        state = f"still {job.state}" if job else "state unknown"
        return JobTimeoutError(f"Job {job_id} was {state} after {timeout:g}s. It keeps running on "
                               f"the server; wait again with wait_for_job({job_id!r}).",
                               job_id=job_id, job=job)

    @staticmethod
    def _progress_key(job: "m.Job"):
        return (job.state, job.queue_position, job.stage)

    @staticmethod
    def _cancelled(job: "m.Job") -> DocExplainerError:
        return JobCancelledError(f"The job for {job.filename} was cancelled.", code="job_cancelled")

    def _chat_params(self, doc_id: str, text: str, kind: str, section_id: int, session_id: Optional[str],
                     level: Optional[str]) -> Dict[str, Any]:
        if kind not in ("ask", "explain", "summarize"):
            raise ValueError("kind must be 'ask', 'explain' or 'summarize'")
        level = getattr(level, "value", level)
        return dict(doc_id=doc_id, kind=kind, text=text, section_id=section_id, session_id=session_id, level=level)

    def _stream_spec(self, params: Dict[str, Any]):
        return self._proto.build(OPERATIONS["chat_stream"], params)

    @staticmethod
    def _events_of(result: "m.ChatResult") -> List["m.StreamEvent"]:
        """A turn that was not streamed, replayed as events (JSON-RPC cannot stream)."""
        return [m.StreamEvent(type="session", data={"id": result.session_id}),
                m.StreamEvent(type="delta", data={"text": result.text}),
                m.StreamEvent(type="done", data=result.model_dump(mode="json"))]

    def _sse_event(self, name: str, payload: str) -> "m.StreamEvent":
        try:
            data = json.loads(payload)
        except ValueError:
            raise ProtocolError("The server sent a malformed stream event.", code="protocol_error")
        if name == "error":
            raise error_from_code(str(data.get("code", "error")), str(data.get("message", "The reply failed.")))
        return m.StreamEvent(type=name, data=data)


class DocExplainerClient(_Base):
    """Blocking client."""

    def __init__(self, base_url: Optional[str] = None, *, protocol: Optional[str] = None,
                 api_key: Optional[str] = None, user_id: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT, http_client: Optional[httpx.Client] = None):
        super().__init__(base_url, protocol, api_key, user_id, timeout)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "DocExplainerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _call(self, op: str, **params: Any) -> Any:
        spec = self._prepare(op, params)
        try:
            response = self._http.request(spec.method, spec.url, **self._kwargs(spec))
        except httpx.HTTPError as e:
            raise self._unavailable(e) from e
        return self._proto.parse(response, spec.request_id)

    # ---- API
    def health(self) -> m.Health:
        return m.Health.model_validate(self._call("health"))

    def start_registration(self, source: Union[str, Path, bytes], *, filename: Optional[str] = None,
                           build_graph: bool = True) -> m.Job:
        """Upload a document and return immediately with the background ``Job``."""
        name, content = self._upload(source, filename)
        return m.Job.model_validate(self._call(
            "register", filename=name, content=content, build_graph=build_graph))

    def get_job(self, job_id: str) -> m.Job:
        return m.Job.model_validate(self._call("get_job", job_id=job_id))

    def list_jobs(self) -> List[m.Job]:
        return [m.Job.model_validate(j) for j in self._call("list_jobs")["jobs"]]

    def wait_for_job(self, job_id: str, *, timeout: Optional[float] = None,
                     poll_interval: float = 0.5, max_poll_interval: float = 5.0,
                     on_update: Optional[Callable[[m.Job], None]] = None,
                     max_poll_errors: int = 3) -> m.Job:
        """Poll until the job finishes and return it. Raises the matching error if it failed.

        ``timeout=None`` waits as long as it takes (registration can take many minutes). Polling
        backs off from ``poll_interval`` to ``max_poll_interval``. ``on_update`` is called whenever
        the state or queue position changes. Up to ``max_poll_errors`` consecutive "server
        unreachable or queue-full" answers are tolerated, so a brief restart does not abort a long wait.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        interval, errors, last, job = poll_interval, 0, None, None
        while True:
            try:
                job, errors = self.get_job(job_id), 0
            except ServerUnavailableError:
                errors += 1
                if errors > max_poll_errors:
                    raise
            else:
                if on_update and self._progress_key(job) != last:
                    on_update(job)
                last = self._progress_key(job)
                if job.done:
                    if job.state == "failed":
                        raise self._failed(job)
                    if job.state == "cancelled":
                        raise self._cancelled(job)
                    return job
            if deadline is not None and time.monotonic() >= deadline:
                raise self._timed_out(job_id, job, timeout)
            time.sleep(interval if deadline is None else max(0.0, min(interval, deadline - time.monotonic())))
            interval = min(interval * 1.5, max_poll_interval)

    def register_document(self, source: Union[str, Path, bytes], *, filename: Optional[str] = None,
                          build_graph: bool = True, timeout: Optional[float] = None,
                          poll_interval: float = 0.5,
                          on_update: Optional[Callable[[m.Job], None]] = None) -> m.Document:
        """Upload a document and wait until the server has processed it.

        Identical content that is already registered returns its ``Document`` straight away.
        Raises ``JobTimeoutError`` (job keeps running) if ``timeout`` seconds pass first.
        """
        job = self.start_registration(source, filename=filename, build_graph=build_graph)
        if on_update:
            on_update(job)
        if not job.done:
            job = self.wait_for_job(job.job_id, timeout=timeout, poll_interval=poll_interval,
                                    on_update=on_update)
        elif job.state == "failed":
            raise self._failed(job)
        elif job.state == "cancelled":
            raise self._cancelled(job)
        return job.document

    def list_documents(self) -> List[m.Document]:
        return [m.Document.model_validate(d) for d in self._call("list_documents")["documents"]]

    def get_document(self, doc_id: str) -> m.Document:
        return m.Document.model_validate(self._call("get_document", doc_id=doc_id))

    def get_section_id(self, doc_id: str, page: int, position: int) -> int:
        return self._section(self._call("section_at", doc_id=doc_id, page=page, position=position))

    def get_context(self, doc_id: str, section_id: int = 0) -> m.Context:
        return m.Context.model_validate(self._call("context", doc_id=doc_id, section_id=section_id))

    def explain(self, doc_id: str, text: str, *, section_id: int = 0,
                style: Optional[Style] = None) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(self._call(
            "explain", doc_id=doc_id, selected_text=text, section_id=section_id,
            style=self._style(style)))

    def summarize(self, doc_id: str, text: str, *, section_id: int = 0) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(self._call(
            "summarize", doc_id=doc_id, selected_text=text, section_id=section_id))

    def ask(self, doc_id: str, question: str, *, section_id: int = 0) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(self._call(
            "ask", doc_id=doc_id, question=question, section_id=section_id))

    def delete_document(self, doc_id: str) -> None:
        self._call("delete_document", doc_id=doc_id)

    # ---- job control
    def pause_job(self, job_id: str) -> m.Job:
        """Stop at the next step and keep the work. A paused job stays on the server (also across a restart)."""
        return m.Job.model_validate(self._call("pause_job", job_id=job_id))

    def resume_job(self, job_id: str) -> m.Job:
        return m.Job.model_validate(self._call("resume_job", job_id=job_id))

    def cancel_job(self, job_id: str) -> m.Job:
        """Stop and discard the work."""
        return m.Job.model_validate(self._call("cancel_job", job_id=job_id))

    # ---- the conversation
    def chat(self, doc_id: str, text: str, *, kind: str = "ask", section_id: int = 0, session_id: Optional[str] = None,
             level: Optional[str] = None) -> m.ChatResult:
        """One turn, answered in one piece and recorded in a session (the one used last unless ``session_id`` is given)."""
        return m.ChatResult.model_validate(self._call("chat", **self._chat_params(doc_id, text, kind, section_id, session_id, level)))

    def chat_stream(self, doc_id: str, text: str, *, kind: str = "ask", section_id: int = 0,
                    session_id: Optional[str] = None, level: Optional[str] = None) -> Iterator[m.StreamEvent]:
        """One turn as it happens: ``session``, ``status`` (what the tutor is doing), ``delta`` (text as it is written),
        then ``done`` (the complete ``ChatResult``). Over JSON-RPC, which cannot stream, the same events arrive at once."""
        params = self._chat_params(doc_id, text, kind, section_id, session_id, level)
        if self.protocol != "rest":
            yield from self._events_of(self.chat(doc_id, text, kind=kind, section_id=section_id, session_id=session_id, level=level))
            return
        spec = self._stream_spec(params)
        try:
            with self._http.stream(spec.method, spec.url, json=spec.json, headers=self._headers, timeout=self._timeout) as response:
                if not response.is_success:
                    response.read()
                    self._proto.parse(response)
                name = ""
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        name = line[7:].strip()
                    elif line.startswith("data: "):
                        yield self._sse_event(name, line[6:])
        except httpx.HTTPError as e:
            raise self._unavailable(e) from e

    def list_sessions(self, doc_id: str) -> List[m.Session]:
        return [m.Session.model_validate(s) for s in self._call("list_sessions", doc_id=doc_id)["sessions"]]

    def create_session(self, doc_id: str, title: str = "") -> m.Session:
        return m.Session.model_validate(self._call("create_session", doc_id=doc_id, title=title))

    def get_session(self, doc_id: str, session_id: str) -> m.SessionDetail:
        """The conversation so far, with its notes and summaries."""
        return m.SessionDetail.model_validate(self._call("get_session", doc_id=doc_id, session_id=session_id))

    def rename_session(self, doc_id: str, session_id: str, title: str) -> m.Session:
        return m.Session.model_validate(self._call("rename_session", doc_id=doc_id, session_id=session_id, title=title))

    def delete_session(self, doc_id: str, session_id: str) -> None:
        self._call("delete_session", doc_id=doc_id, session_id=session_id)

    def add_note(self, doc_id: str, session_id: str, text: str, kind: str = "note") -> m.Note:
        return m.Note.model_validate(self._call("add_note", doc_id=doc_id, session_id=session_id, text=text, kind=kind))

    def edit_note(self, doc_id: str, session_id: str, note_id: str, text: str) -> None:
        self._call("edit_note", doc_id=doc_id, session_id=session_id, note_id=note_id, text=text)

    def delete_note(self, doc_id: str, session_id: str, note_id: str) -> None:
        self._call("delete_note", doc_id=doc_id, session_id=session_id, note_id=note_id)

    def summarize_session(self, doc_id: str, session_id: str) -> m.Note:
        """Have the tutor write a summary of the conversation; it is kept as a note of the session."""
        return m.Note.model_validate(self._call("summarize_session", doc_id=doc_id, session_id=session_id))

    # ---- the document's structure and knowledge
    def get_structure(self, doc_id: str) -> m.Structure:
        return m.Structure.model_validate(self._call("structure", doc_id=doc_id))

    def get_knowledge(self, doc_id: str, upto: Optional[int] = None) -> m.Knowledge:
        """The knowledge graph as a reader at reading position ``upto`` would know it (default: all of it)."""
        return m.Knowledge.model_validate(self._call("knowledge", doc_id=doc_id, upto=upto))

    def get_section_knowledge(self, doc_id: str, section_id: int) -> m.SectionKnowledge:
        return m.SectionKnowledge.model_validate(self._call("section_knowledge", doc_id=doc_id, section_id=section_id))

    def delete_knowledge(self, doc_id: str) -> None:
        """Drop the knowledge graph (and this reader's tracking) of a document; the document stays readable."""
        self._call("delete_knowledge", doc_id=doc_id)

    def get_review(self, doc_id: str, limit: int = 20) -> List[m.ReviewItem]:
        """Concepts you are likely forgetting, weakest first (spaced repetition)."""
        return [m.ReviewItem.model_validate(i) for i in (self._call("review", doc_id=doc_id, limit=limit))["items"]]

    def get_history(self, doc_id: str, limit: int = 200) -> List[m.HistoryPoint]:
        """How your belief in each concept changed, oldest first."""
        return [m.HistoryPoint.model_validate(i) for i in (self._call("history", doc_id=doc_id, limit=limit))["points"]]

    def export_session(self, doc_id: str, session_id: str) -> m.Export:
        """A conversation and its notes as Markdown."""
        return m.Export.model_validate(self._call("export_session", doc_id=doc_id, session_id=session_id))

    def get_growth(self, doc_id: str) -> List[m.GrowthPoint]:
        return [m.GrowthPoint.model_validate(p) for p in self._call("growth", doc_id=doc_id)["points"]]

    def get_prerequisites(self, doc_id: str, concept: str, max_depth: int = 3) -> m.ConceptResult:
        return m.ConceptResult.model_validate(self._call("prerequisites", doc_id=doc_id, name=concept, max_depth=max_depth))

    def get_learning_path(self, doc_id: str, concept: str) -> m.ConceptResult:
        return m.ConceptResult.model_validate(self._call("learning_path", doc_id=doc_id, name=concept))

    def recommend_concepts(self, doc_id: str, concept: str, limit: int = 5) -> m.ConceptResult:
        return m.ConceptResult.model_validate(self._call("recommendations", doc_id=doc_id, name=concept, limit=limit))

    # ---- the learner
    def get_learner(self, doc_id: str) -> m.Learner:
        return m.Learner.model_validate(self._call("learner", doc_id=doc_id))

    def start_diagnostic(self, doc_id: str, level: str = "intermediate", known_topics: Sequence[str] = (),
                         questions: int = 6) -> m.Diagnostic:
        """Say where you stand (``level``, topics you know) and get a few questions that pin it down."""
        return m.Diagnostic.model_validate(self._call("diagnostic_start", doc_id=doc_id, level=level,
                                                      known_topics=list(known_topics), questions=questions))

    def finish_diagnostic(self, doc_id: str) -> m.Diagnostic:
        return m.Diagnostic.model_validate(self._call("diagnostic_finish", doc_id=doc_id))

    def practice_quiz(self, doc_id: str, section_id: Optional[int] = None, count: int = 5) -> List[m.QuizQuestion]:
        return [m.QuizQuestion.model_validate(q) for q in self._call("quiz_new", doc_id=doc_id, section_id=section_id, count=count)["questions"]]

    def answer_quiz(self, doc_id: str, question_id: str, choice: int) -> m.AnswerResult:
        return m.AnswerResult.model_validate(self._call("quiz_answer", doc_id=doc_id, question_id=question_id, choice=choice))

    def send_feedback(self, doc_id: str, verdict: str, concept_ids: Sequence[str]) -> None:
        """``verdict``: "too_easy", "too_hard" or "just_right"; concept ids are in ``explanation.context_used["concept_ids"]``."""
        self._call("feedback", doc_id=doc_id, verdict=verdict, concept_ids=list(concept_ids))

    def mark_progress(self, doc_id: str, section_id: int) -> None:
        """The reader finished this section."""
        self._call("progress", doc_id=doc_id, section_id=section_id)


class AsyncDocExplainerClient(_Base):
    """Same API as ``DocExplainerClient``, awaitable. Use ``async with`` or call ``await aclose()``."""

    def __init__(self, base_url: Optional[str] = None, *, protocol: Optional[str] = None,
                 api_key: Optional[str] = None, user_id: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT, http_client: Optional[httpx.AsyncClient] = None):
        super().__init__(base_url, protocol, api_key, user_id, timeout)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncDocExplainerClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _call(self, op: str, **params: Any) -> Any:
        spec = self._prepare(op, params)
        try:
            response = await self._http.request(spec.method, spec.url, **self._kwargs(spec))
        except httpx.HTTPError as e:
            raise self._unavailable(e) from e
        return self._proto.parse(response, spec.request_id)

    async def health(self) -> m.Health:
        return m.Health.model_validate(await self._call("health"))

    async def start_registration(self, source: Union[str, Path, bytes], *,
                                 filename: Optional[str] = None, build_graph: bool = True) -> m.Job:
        name, content = self._upload(source, filename)
        return m.Job.model_validate(await self._call(
            "register", filename=name, content=content, build_graph=build_graph))

    async def get_job(self, job_id: str) -> m.Job:
        return m.Job.model_validate(await self._call("get_job", job_id=job_id))

    async def list_jobs(self) -> List[m.Job]:
        return [m.Job.model_validate(j) for j in (await self._call("list_jobs"))["jobs"]]

    async def wait_for_job(self, job_id: str, *, timeout: Optional[float] = None,
                           poll_interval: float = 0.5, max_poll_interval: float = 5.0,
                           on_update: Optional[Callable[[m.Job], None]] = None,
                           max_poll_errors: int = 3) -> m.Job:
        """See ``DocExplainerClient.wait_for_job``. Cancelling the task stops waiting only; the job
        keeps running on the server."""
        deadline = None if timeout is None else time.monotonic() + timeout
        interval, errors, last, job = poll_interval, 0, None, None
        while True:
            try:
                job, errors = await self.get_job(job_id), 0
            except ServerUnavailableError:
                errors += 1
                if errors > max_poll_errors:
                    raise
            else:
                if on_update and self._progress_key(job) != last:
                    on_update(job)
                last = self._progress_key(job)
                if job.done:
                    if job.state == "failed":
                        raise self._failed(job)
                    if job.state == "cancelled":
                        raise self._cancelled(job)
                    return job
            if deadline is not None and time.monotonic() >= deadline:
                raise self._timed_out(job_id, job, timeout)
            await asyncio.sleep(interval if deadline is None else max(0.0, min(interval, deadline - time.monotonic())))
            interval = min(interval * 1.5, max_poll_interval)

    async def register_document(self, source: Union[str, Path, bytes], *,
                                filename: Optional[str] = None, build_graph: bool = True,
                                timeout: Optional[float] = None, poll_interval: float = 0.5,
                                on_update: Optional[Callable[[m.Job], None]] = None) -> m.Document:
        job = await self.start_registration(source, filename=filename, build_graph=build_graph)
        if on_update:
            on_update(job)
        if not job.done:
            job = await self.wait_for_job(job.job_id, timeout=timeout, poll_interval=poll_interval,
                                          on_update=on_update)
        elif job.state == "failed":
            raise self._failed(job)
        elif job.state == "cancelled":
            raise self._cancelled(job)
        return job.document

    async def list_documents(self) -> List[m.Document]:
        return [m.Document.model_validate(d) for d in (await self._call("list_documents"))["documents"]]

    async def get_document(self, doc_id: str) -> m.Document:
        return m.Document.model_validate(await self._call("get_document", doc_id=doc_id))

    async def get_section_id(self, doc_id: str, page: int, position: int) -> int:
        return self._section(await self._call("section_at", doc_id=doc_id, page=page, position=position))

    async def get_context(self, doc_id: str, section_id: int = 0) -> m.Context:
        return m.Context.model_validate(await self._call("context", doc_id=doc_id, section_id=section_id))

    async def explain(self, doc_id: str, text: str, *, section_id: int = 0,
                      style: Optional[Style] = None) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(await self._call(
            "explain", doc_id=doc_id, selected_text=text, section_id=section_id,
            style=self._style(style)))

    async def summarize(self, doc_id: str, text: str, *, section_id: int = 0) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(await self._call(
            "summarize", doc_id=doc_id, selected_text=text, section_id=section_id))

    async def ask(self, doc_id: str, question: str, *, section_id: int = 0) -> m.ExplanationResult:
        return m.ExplanationResult.model_validate(await self._call(
            "ask", doc_id=doc_id, question=question, section_id=section_id))

    async def delete_document(self, doc_id: str) -> None:
        await self._call("delete_document", doc_id=doc_id)

    # ---- job control
    async def pause_job(self, job_id: str) -> m.Job:
        """Stop at the next step and keep the work. A paused job stays on the server (also across a restart)."""
        return m.Job.model_validate(await self._call("pause_job", job_id=job_id))

    async def resume_job(self, job_id: str) -> m.Job:
        return m.Job.model_validate(await self._call("resume_job", job_id=job_id))

    async def cancel_job(self, job_id: str) -> m.Job:
        """Stop and discard the work."""
        return m.Job.model_validate(await self._call("cancel_job", job_id=job_id))

    # ---- the conversation
    async def chat(self, doc_id: str, text: str, *, kind: str = "ask", section_id: int = 0, session_id: Optional[str] = None,
             level: Optional[str] = None) -> m.ChatResult:
        """One turn, answered in one piece and recorded in a session (the one used last unless ``session_id`` is given)."""
        return m.ChatResult.model_validate(await self._call("chat", **self._chat_params(doc_id, text, kind, section_id, session_id, level)))

    async def chat_stream(self, doc_id: str, text: str, *, kind: str = "ask", section_id: int = 0,
                          session_id: Optional[str] = None, level: Optional[str] = None) -> AsyncIterator[m.StreamEvent]:
        """See ``DocExplainerClient.chat_stream``."""
        params = self._chat_params(doc_id, text, kind, section_id, session_id, level)
        if self.protocol != "rest":
            for ev in self._events_of(await self.chat(doc_id, text, kind=kind, section_id=section_id, session_id=session_id, level=level)):
                yield ev
            return
        spec = self._stream_spec(params)
        try:
            async with self._http.stream(spec.method, spec.url, json=spec.json, headers=self._headers, timeout=self._timeout) as response:
                if not response.is_success:
                    await response.aread()
                    self._proto.parse(response)
                name = ""
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        name = line[7:].strip()
                    elif line.startswith("data: "):
                        yield self._sse_event(name, line[6:])
        except httpx.HTTPError as e:
            raise self._unavailable(e) from e

    async def list_sessions(self, doc_id: str) -> List[m.Session]:
        return [m.Session.model_validate(s) for s in (await self._call("list_sessions", doc_id=doc_id))["sessions"]]

    async def create_session(self, doc_id: str, title: str = "") -> m.Session:
        return m.Session.model_validate(await self._call("create_session", doc_id=doc_id, title=title))

    async def get_session(self, doc_id: str, session_id: str) -> m.SessionDetail:
        """The conversation so far, with its notes and summaries."""
        return m.SessionDetail.model_validate(await self._call("get_session", doc_id=doc_id, session_id=session_id))

    async def rename_session(self, doc_id: str, session_id: str, title: str) -> m.Session:
        return m.Session.model_validate(await self._call("rename_session", doc_id=doc_id, session_id=session_id, title=title))

    async def delete_session(self, doc_id: str, session_id: str) -> None:
        await self._call("delete_session", doc_id=doc_id, session_id=session_id)

    async def add_note(self, doc_id: str, session_id: str, text: str, kind: str = "note") -> m.Note:
        return m.Note.model_validate(await self._call("add_note", doc_id=doc_id, session_id=session_id, text=text, kind=kind))

    async def edit_note(self, doc_id: str, session_id: str, note_id: str, text: str) -> None:
        await self._call("edit_note", doc_id=doc_id, session_id=session_id, note_id=note_id, text=text)

    async def delete_note(self, doc_id: str, session_id: str, note_id: str) -> None:
        await self._call("delete_note", doc_id=doc_id, session_id=session_id, note_id=note_id)

    async def summarize_session(self, doc_id: str, session_id: str) -> m.Note:
        """Have the tutor write a summary of the conversation; it is kept as a note of the session."""
        return m.Note.model_validate(await self._call("summarize_session", doc_id=doc_id, session_id=session_id))

    # ---- the document's structure and knowledge
    async def get_structure(self, doc_id: str) -> m.Structure:
        return m.Structure.model_validate(await self._call("structure", doc_id=doc_id))

    async def get_knowledge(self, doc_id: str, upto: Optional[int] = None) -> m.Knowledge:
        """The knowledge graph as a reader at reading position ``upto`` would know it (default: all of it)."""
        return m.Knowledge.model_validate(await self._call("knowledge", doc_id=doc_id, upto=upto))

    async def get_section_knowledge(self, doc_id: str, section_id: int) -> m.SectionKnowledge:
        return m.SectionKnowledge.model_validate(await self._call("section_knowledge", doc_id=doc_id, section_id=section_id))

    async def delete_knowledge(self, doc_id: str) -> None:
        """Drop the knowledge graph (and this reader's tracking) of a document; the document stays readable."""
        await self._call("delete_knowledge", doc_id=doc_id)

    async def get_review(self, doc_id: str, limit: int = 20) -> List[m.ReviewItem]:
        """Concepts you are likely forgetting, weakest first (spaced repetition)."""
        return [m.ReviewItem.model_validate(i) for i in (await self._call("review", doc_id=doc_id, limit=limit))["items"]]

    async def get_history(self, doc_id: str, limit: int = 200) -> List[m.HistoryPoint]:
        """How your belief in each concept changed, oldest first."""
        return [m.HistoryPoint.model_validate(i) for i in (await self._call("history", doc_id=doc_id, limit=limit))["points"]]

    async def export_session(self, doc_id: str, session_id: str) -> m.Export:
        """A conversation and its notes as Markdown."""
        return m.Export.model_validate(await self._call("export_session", doc_id=doc_id, session_id=session_id))

    async def get_growth(self, doc_id: str) -> List[m.GrowthPoint]:
        return [m.GrowthPoint.model_validate(p) for p in (await self._call("growth", doc_id=doc_id))["points"]]

    async def get_prerequisites(self, doc_id: str, concept: str, max_depth: int = 3) -> m.ConceptResult:
        return m.ConceptResult.model_validate(await self._call("prerequisites", doc_id=doc_id, name=concept, max_depth=max_depth))

    async def get_learning_path(self, doc_id: str, concept: str) -> m.ConceptResult:
        return m.ConceptResult.model_validate(await self._call("learning_path", doc_id=doc_id, name=concept))

    async def recommend_concepts(self, doc_id: str, concept: str, limit: int = 5) -> m.ConceptResult:
        return m.ConceptResult.model_validate(await self._call("recommendations", doc_id=doc_id, name=concept, limit=limit))

    # ---- the learner
    async def get_learner(self, doc_id: str) -> m.Learner:
        return m.Learner.model_validate(await self._call("learner", doc_id=doc_id))

    async def start_diagnostic(self, doc_id: str, level: str = "intermediate", known_topics: Sequence[str] = (),
                         questions: int = 6) -> m.Diagnostic:
        """Say where you stand (``level``, topics you know) and get a few questions that pin it down."""
        return m.Diagnostic.model_validate(await self._call("diagnostic_start", doc_id=doc_id, level=level,
                                                      known_topics=list(known_topics), questions=questions))

    async def finish_diagnostic(self, doc_id: str) -> m.Diagnostic:
        return m.Diagnostic.model_validate(await self._call("diagnostic_finish", doc_id=doc_id))

    async def practice_quiz(self, doc_id: str, section_id: Optional[int] = None, count: int = 5) -> List[m.QuizQuestion]:
        return [m.QuizQuestion.model_validate(q) for q in (await self._call("quiz_new", doc_id=doc_id, section_id=section_id, count=count))["questions"]]

    async def answer_quiz(self, doc_id: str, question_id: str, choice: int) -> m.AnswerResult:
        return m.AnswerResult.model_validate(await self._call("quiz_answer", doc_id=doc_id, question_id=question_id, choice=choice))

    async def send_feedback(self, doc_id: str, verdict: str, concept_ids: Sequence[str]) -> None:
        """``verdict``: "too_easy", "too_hard" or "just_right"; concept ids are in ``explanation.context_used["concept_ids"]``."""
        await self._call("feedback", doc_id=doc_id, verdict=verdict, concept_ids=list(concept_ids))

    async def mark_progress(self, doc_id: str, section_id: int) -> None:
        """The reader finished this section."""
        await self._call("progress", doc_id=doc_id, section_id=section_id)
