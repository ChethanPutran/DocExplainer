"""ChatService: sessions of a document, their history, and notes/summaries.

It knows nothing about how answers are produced; the caller records each exchange. Summaries are written by the
model through the ``TextLLM`` port from the conversation itself (never from anything else).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..common.ports import LLMError, TextLLM
from .store import DEFAULT_TITLE, Message, Note, Session, SqliteChatStore

MAX_TITLE = 60
MAX_NOTE = 20_000
_SUMMARY_SYSTEM = ("You write study notes from a conversation between a reader and a tutor about one document. "
                   "Use ONLY what is in the conversation. Write short Markdown: a one-line gist, then bullet points of the key "
                   "ideas, then any open questions the reader still had. No preamble.")


class ChatError(ValueError):
    """The session, note or request is not valid (the API turns this into a clear error)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ChatService:
    def __init__(self, store: SqliteChatStore, llm: Optional[TextLLM] = None, clock=time.time):
        self.store, self.llm, self.clock = store, llm, clock

    # ------------------------------------------------------------------ sessions
    def create_session(self, user_id: str, doc_id: str, title: str = "") -> Session:
        return self.store.create_session(user_id, doc_id, (title or "").strip()[:MAX_TITLE] or DEFAULT_TITLE, self.clock())

    def list_sessions(self, user_id: str, doc_id: str) -> List[Session]:
        return self.store.list_sessions(user_id, doc_id)

    def session(self, user_id: str, doc_id: str, session_id: str) -> Session:
        s = self.store.get_session(user_id, doc_id, session_id)
        if s is None:   # unknown and someone else's look the same on purpose
            raise ChatError("session_not_found", f"Session {session_id!r} not found.")
        return s

    def open_session(self, user_id: str, doc_id: str, session_id: Optional[str] = None) -> Session:
        """The named session, else the one used last, else a new one."""
        if session_id:
            return self.session(user_id, doc_id, session_id)
        existing = self.store.list_sessions(user_id, doc_id)
        return existing[0] if existing else self.create_session(user_id, doc_id)

    def rename(self, user_id: str, doc_id: str, session_id: str, title: str) -> Session:
        self.session(user_id, doc_id, session_id)
        title = (title or "").strip()[:MAX_TITLE]
        if not title:
            raise ChatError("invalid_params", "A session needs a title.")
        self.store.rename_session(session_id, title)
        return self.session(user_id, doc_id, session_id)

    def delete(self, user_id: str, doc_id: str, session_id: str) -> None:
        self.session(user_id, doc_id, session_id)
        self.store.delete_session(session_id)

    def forget_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)

    # ------------------------------------------------------------------ the conversation
    def history(self, user_id: str, doc_id: str, session_id: str) -> Tuple[Session, List[Message], List[Note]]:
        s = self.session(user_id, doc_id, session_id)
        return s, self.store.messages(session_id), self.store.notes(session_id)

    def record_question(self, session: Session, text: str, kind: str, meta: Optional[Dict[str, Any]] = None) -> Message:
        m = self.store.add_message(session.id, "user", text, kind, meta or {}, self.clock())
        if session.title == DEFAULT_TITLE and not session.messages:       # name a fresh session after its first question
            first = " ".join(text.split())
            if first:
                self.store.rename_session(session.id, first[:MAX_TITLE])
        return m

    def record_answer(self, session: Session, text: str, kind: str, meta: Optional[Dict[str, Any]] = None) -> Message:
        return self.store.add_message(session.id, "assistant", text, kind, meta or {}, self.clock())

    # ------------------------------------------------------------------ notes and summaries
    def add_note(self, user_id: str, doc_id: str, session_id: str, text: str, kind: str = "note") -> Note:
        self.session(user_id, doc_id, session_id)
        text = (text or "").strip()
        if not text:
            raise ChatError("invalid_params", "A note cannot be empty.")
        if kind not in ("note", "summary"):
            raise ChatError("invalid_params", "kind must be 'note' or 'summary'.")
        return self.store.add_note(session_id, kind, text[:MAX_NOTE], self.clock())

    def edit_note(self, user_id: str, doc_id: str, session_id: str, note_id: str, text: str) -> None:
        self.session(user_id, doc_id, session_id)
        text = (text or "").strip()
        if not text:
            raise ChatError("invalid_params", "A note cannot be empty.")
        if not self.store.update_note(session_id, note_id, text[:MAX_NOTE]):
            raise ChatError("note_not_found", f"Note {note_id!r} not found.")

    def delete_note(self, user_id: str, doc_id: str, session_id: str, note_id: str) -> None:
        self.session(user_id, doc_id, session_id)
        if not self.store.delete_note(session_id, note_id):
            raise ChatError("note_not_found", f"Note {note_id!r} not found.")

    def summarize(self, user_id: str, doc_id: str, session_id: str) -> Note:
        """Write a summary of the session as a note. Needs at least one exchange."""
        _, messages, _ = self.history(user_id, doc_id, session_id)
        if not any(m.role == "assistant" for m in messages):
            raise ChatError("nothing_to_summarize", "Ask something first: there is no conversation to summarise yet.")
        if self.llm is None:
            raise ChatError("not_ready", "No language model is available to write the summary.")
        transcript = "\n\n".join(f"{'READER' if m.role == 'user' else 'TUTOR'}: {m.text}" for m in messages)[-12_000:]
        try:
            text = "".join(self.llm.stream_text(_SUMMARY_SYSTEM, transcript)).strip()
        except LLMError:
            raise
        if not text:
            raise ChatError("empty_summary", "The model returned an empty summary. Try again.")
        return self.store.add_note(session_id, "summary", text[:MAX_NOTE], self.clock())
