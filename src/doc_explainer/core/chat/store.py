"""Conversations: sessions of one document, their messages, and the notes/summaries kept with them.

Everything is per user AND per document. Messages keep what the reader asked and what the tutor answered
(with the follow-up questions and where in the document it was), so a past session can be read again.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, title TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user_id, doc_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
    text TEXT NOT NULL, kind TEXT NOT NULL, meta TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);
CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
    created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS notes_session ON notes (session_id, created_at);
"""
DEFAULT_TITLE = "New session"


@dataclass
class Session:
    id: str
    doc_id: str
    title: str
    created_at: float
    updated_at: float
    messages: int = 0
    notes: int = 0


@dataclass
class Message:
    id: int
    session_id: str
    role: str                   # user | assistant
    text: str
    kind: str                   # ask | explain | summarize
    created_at: float
    meta: Dict[str, Any] = field(default_factory=dict)   # follow_ups, section, concept_ids, resources, ...


@dataclass
class Note:
    id: str
    session_id: str
    kind: str                   # note (written by the reader) | summary (written by the tutor)
    text: str
    created_at: float


class SqliteChatStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ sessions
    def create_session(self, user_id: str, doc_id: str, title: str, at: float) -> Session:
        sid = uuid.uuid4().hex
        with self._lock, self._db:
            self._db.execute("INSERT INTO sessions (id, user_id, doc_id, title, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                             (sid, user_id, doc_id, title, at, at))
        return Session(sid, doc_id, title, at, at)

    _SESSION_SQL = ("SELECT s.id, s.doc_id, s.title, s.created_at, s.updated_at, "
                    "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id), (SELECT COUNT(*) FROM notes n WHERE n.session_id = s.id) "
                    "FROM sessions s ")

    def get_session(self, user_id: str, doc_id: str, session_id: str) -> Optional[Session]:
        """None unless this user owns this session of this document."""
        with self._lock:
            r = self._db.execute(self._SESSION_SQL + "WHERE s.id=? AND s.user_id=? AND s.doc_id=?", (session_id, user_id, doc_id)).fetchone()
        return Session(*r) if r else None

    def list_sessions(self, user_id: str, doc_id: str) -> List[Session]:
        with self._lock:
            rows = self._db.execute(self._SESSION_SQL + "WHERE s.user_id=? AND s.doc_id=? ORDER BY s.updated_at DESC, s.created_at DESC",
                                    (user_id, doc_id)).fetchall()
        return [Session(*r) for r in rows]

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))

    def touch(self, session_id: str, at: float) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (at, session_id))

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._db.execute("DELETE FROM notes WHERE session_id=?", (session_id,))
            self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def delete_document(self, doc_id: str) -> None:
        """Forget every session of a document (all users)."""
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM sessions WHERE doc_id=?", (doc_id,))]
        for sid in ids:
            self.delete_session(sid)

    # ------------------------------------------------------------------ messages
    def add_message(self, session_id: str, role: str, text: str, kind: str, meta: Dict[str, Any], at: float) -> Message:
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO messages (session_id, role, text, kind, meta, created_at) VALUES (?,?,?,?,?,?)",
                                   (session_id, role, text, kind, json.dumps(meta), at))
            self._db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (at, session_id))
        return Message(cur.lastrowid, session_id, role, text, kind, at, meta)

    def messages(self, session_id: str) -> List[Message]:
        with self._lock:
            rows = self._db.execute("SELECT id, session_id, role, text, kind, created_at, meta FROM messages WHERE session_id=? ORDER BY id",
                                    (session_id,)).fetchall()
        return [Message(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6])) for r in rows]

    # ------------------------------------------------------------------ notes
    def add_note(self, session_id: str, kind: str, text: str, at: float) -> Note:
        nid = uuid.uuid4().hex
        with self._lock, self._db:
            self._db.execute("INSERT INTO notes (id, session_id, kind, text, created_at) VALUES (?,?,?,?,?)", (nid, session_id, kind, text, at))
            self._db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (at, session_id))
        return Note(nid, session_id, kind, text, at)

    def notes(self, session_id: str) -> List[Note]:
        with self._lock:
            rows = self._db.execute("SELECT id, session_id, kind, text, created_at FROM notes WHERE session_id=? ORDER BY created_at, rowid",
                                    (session_id,)).fetchall()
        return [Note(*r) for r in rows]

    def update_note(self, session_id: str, note_id: str, text: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("UPDATE notes SET text=? WHERE id=? AND session_id=?", (text, note_id, session_id)).rowcount == 1

    def delete_note(self, session_id: str, note_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM notes WHERE id=? AND session_id=?", (note_id, session_id)).rowcount == 1
