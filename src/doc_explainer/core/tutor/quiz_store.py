"""Quiz questions that have been issued, with their answers kept server-side until the learner answers."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_items (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, concept_id TEXT NOT NULL,
    kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL, answered INTEGER NOT NULL DEFAULT 0, chosen INTEGER, correct INTEGER);
CREATE INDEX IF NOT EXISTS quiz_user ON quiz_items (user_id, doc_id, kind);
"""


class SqliteQuizStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def add(self, item_id: str, user_id: str, doc_id: str, concept_id: str, kind: str, payload: dict, at: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO quiz_items (id, user_id, doc_id, concept_id, kind, payload, created_at) VALUES (?,?,?,?,?,?,?)",
                             (item_id, user_id, doc_id, concept_id, kind, json.dumps(payload), at))

    def get(self, item_id: str, user_id: str, doc_id: str) -> Optional[dict]:
        """None unless this learner was issued this question for this document."""
        with self._lock:
            r = self._db.execute("SELECT concept_id, kind, payload, answered, chosen, correct FROM quiz_items "
                                 "WHERE id=? AND user_id=? AND doc_id=?", (item_id, user_id, doc_id)).fetchone()
        if r is None:
            return None
        return {"id": item_id, "concept_id": r[0], "kind": r[1], "payload": json.loads(r[2]), "answered": bool(r[3]),
                "chosen": r[4], "correct": None if r[5] is None else bool(r[5])}

    def mark_answered(self, item_id: str, chosen: int, correct: bool) -> bool:
        """Atomically record the answer; False if it had already been answered (so evidence is never counted twice)."""
        with self._lock, self._db:
            cur = self._db.execute("UPDATE quiz_items SET answered=1, chosen=?, correct=? WHERE id=? AND answered=0",
                                   (chosen, int(correct), item_id))
            return cur.rowcount == 1

    def asked_concepts(self, user_id: str, doc_id: str, kind: str) -> list:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT concept_id FROM quiz_items WHERE user_id=? AND doc_id=? AND kind=?",
                                                   (user_id, doc_id, kind))]
