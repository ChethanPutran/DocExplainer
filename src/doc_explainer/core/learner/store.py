"""SQLite store for learner state. Everything is keyed by (user, document): users never see each other's data."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..common.learner import Belief, LearnerProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (user_id TEXT, doc_id TEXT, prior REAL, level TEXT, onboarding TEXT, created_at REAL,
    PRIMARY KEY (user_id, doc_id));
CREATE TABLE IF NOT EXISTS beliefs (user_id TEXT, doc_id TEXT, concept_id TEXT, p_known REAL, evidence INTEGER, updated_at REAL,
    PRIMARY KEY (user_id, doc_id, concept_id));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, doc_id TEXT, at REAL, kind TEXT,
    concept_id TEXT, payload TEXT, p_before REAL, p_after REAL);
CREATE INDEX IF NOT EXISTS events_user ON events (user_id, doc_id, id);
CREATE TABLE IF NOT EXISTS positions (user_id TEXT, doc_id TEXT, ordinal INTEGER, updated_at REAL, PRIMARY KEY (user_id, doc_id));
"""


class SqliteLearnerStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ profile
    def get_profile(self, user_id: str, doc_id: str) -> Optional[LearnerProfile]:
        with self._lock:
            r = self._db.execute("SELECT prior, level, onboarding, created_at FROM profiles WHERE user_id=? AND doc_id=?",
                                 (user_id, doc_id)).fetchone()
            return LearnerProfile(user_id, doc_id, r[0], r[1], json.loads(r[2]), r[3]) if r else None

    def save_profile(self, p: LearnerProfile) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?)",
                             (p.user_id, p.doc_id, p.prior, p.level, json.dumps(p.onboarding), p.created_at))

    # ------------------------------------------------------------------ beliefs
    def get_beliefs(self, user_id: str, doc_id: str, concept_ids: Optional[Iterable[str]] = None) -> Dict[str, Belief]:
        with self._lock:
            rows = self._db.execute("SELECT concept_id, p_known, evidence, updated_at FROM beliefs WHERE user_id=? AND doc_id=?",
                                    (user_id, doc_id)).fetchall()
        wanted = set(concept_ids) if concept_ids is not None else None
        return {r[0]: Belief(r[1], r[2], r[3]) for r in rows if wanted is None or r[0] in wanted}

    def set_beliefs(self, user_id: str, doc_id: str, beliefs: Dict[str, Belief]) -> None:
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO beliefs VALUES (?,?,?,?,?,?)",
                                 [(user_id, doc_id, cid, b.p_known, b.evidence, b.updated_at) for cid, b in beliefs.items()])

    # ------------------------------------------------------------------ events (the audit trail)
    def add_events(self, user_id: str, doc_id: str, rows: Sequence[tuple]) -> None:
        """rows: (at, kind, concept_id, payload_json, p_before, p_after)"""
        with self._lock, self._db:
            self._db.executemany("INSERT INTO events (user_id, doc_id, at, kind, concept_id, payload, p_before, p_after) "
                                 "VALUES (?,?,?,?,?,?,?,?)", [(user_id, doc_id, *r) for r in rows])

    def events(self, user_id: str, doc_id: str, limit: int = 200) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT at, kind, concept_id, payload, p_before, p_after FROM events "
                                    "WHERE user_id=? AND doc_id=? ORDER BY id DESC LIMIT ?", (user_id, doc_id, limit)).fetchall()
        return [{"at": r[0], "kind": r[1], "concept_id": r[2], "payload": json.loads(r[3]), "before": r[4], "after": r[5]}
                for r in reversed(rows)]

    # ------------------------------------------------------------------ reading position
    def get_position(self, user_id: str, doc_id: str) -> Optional[int]:
        with self._lock:
            r = self._db.execute("SELECT ordinal FROM positions WHERE user_id=? AND doc_id=?", (user_id, doc_id)).fetchone()
            return r[0] if r else None

    def set_position(self, user_id: str, doc_id: str, ordinal: int, at: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?)", (user_id, doc_id, ordinal, at))

    # ------------------------------------------------------------------ privacy
    def delete(self, user_id: str, doc_id: Optional[str] = None) -> None:
        """Forget a learner (for one document, or entirely)."""
        with self._lock, self._db:
            for table in ("profiles", "beliefs", "events", "positions"):
                if doc_id is None:
                    self._db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                else:
                    self._db.execute(f"DELETE FROM {table} WHERE user_id=? AND doc_id=?", (user_id, doc_id))
