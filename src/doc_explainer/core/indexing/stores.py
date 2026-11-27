"""Where the structure and the source text live.

* ``SqliteStructureStore``: the graph store. STRUCTURE ONLY: node ids, kinds, reading order, page
  spans, section titles (labels) and the edges between nodes. No body text. SQLite needs no server and
  answers the questions we ask (children, ancestors, page lookups) directly.
* ``JsonContentStore``: the parsed document with its text, as JSON (never pickle). It is the source of
  truth: the vector index is derived from it and can be rebuilt without re-parsing the file.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..common.structure import DocumentStructure, EdgeKind


@dataclass(frozen=True)
class NodeRow:
    id: str
    kind: str
    parent_id: Optional[str]
    ordinal: int
    depth: int
    page_start: int
    page_end: int
    title: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, title TEXT, source_format TEXT, page_count INTEGER);
CREATE TABLE IF NOT EXISTS nodes (
    doc_id TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, parent_id TEXT, ordinal INTEGER NOT NULL,
    depth INTEGER NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (doc_id, id));
CREATE INDEX IF NOT EXISTS nodes_order ON nodes (doc_id, ordinal);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes (doc_id, parent_id);
CREATE TABLE IF NOT EXISTS edges (doc_id TEXT NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL,
    PRIMARY KEY (doc_id, src, dst, kind));
CREATE INDEX IF NOT EXISTS edges_dst ON edges (doc_id, dst);
"""


class SqliteStructureStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def save(self, structure: DocumentStructure) -> None:
        """Replace this document's structure. Titles are kept (they are labels); body text never is."""
        with self._lock, self._db:
            self._delete(structure.doc_id)
            self._db.execute("INSERT INTO documents VALUES (?,?,?,?)",
                             (structure.doc_id, structure.title, structure.source_format, structure.page_count))
            self._db.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",
                [(structure.doc_id, n.id, n.kind.value, n.parent_id, n.ordinal, n.depth, n.page_start, n.page_end,
                  n.title if n.kind.value in ("section", "document") else "") for n in structure.nodes])
            self._db.executemany("INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                                 [(structure.doc_id, e.src, e.dst, e.kind.value) for e in structure.edges])

    def _delete(self, doc_id: str) -> None:
        for table in ("edges", "nodes", "documents"):
            self._db.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))

    def delete(self, doc_id: str) -> None:
        with self._lock, self._db:
            self._delete(doc_id)

    def has(self, doc_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    @staticmethod
    def _row(r) -> NodeRow:
        return NodeRow(*r)

    _COLS = "id, kind, parent_id, ordinal, depth, page_start, page_end, title"

    def nodes(self, doc_id: str, kind: Optional[str] = None) -> List[NodeRow]:
        with self._lock:
            sql = f"SELECT {self._COLS} FROM nodes WHERE doc_id=?" + (" AND kind=?" if kind else "") + " ORDER BY ordinal"
            return [self._row(r) for r in self._db.execute(sql, (doc_id, kind) if kind else (doc_id,))]

    def node(self, doc_id: str, node_id: str) -> Optional[NodeRow]:
        with self._lock:
            r = self._db.execute(f"SELECT {self._COLS} FROM nodes WHERE doc_id=? AND id=?", (doc_id, node_id)).fetchone()
            return self._row(r) if r else None

    def children(self, doc_id: str, node_id: str) -> List[NodeRow]:
        with self._lock:
            return [self._row(r) for r in self._db.execute(
                f"SELECT {self._COLS} FROM nodes WHERE doc_id=? AND parent_id=? ORDER BY ordinal", (doc_id, node_id))]

    def ancestors(self, doc_id: str, node_id: str) -> List[NodeRow]:
        """From the document root down to (and including) this node."""
        with self._lock:
            rows = self._db.execute(f"""
                WITH RECURSIVE up(id, parent_id, ordinal, kind, depth, page_start, page_end, title) AS (
                    SELECT id, parent_id, ordinal, kind, depth, page_start, page_end, title FROM nodes WHERE doc_id=? AND id=?
                    UNION ALL
                    SELECT n.id, n.parent_id, n.ordinal, n.kind, n.depth, n.page_start, n.page_end, n.title
                    FROM nodes n JOIN up ON n.doc_id=? AND n.id = up.parent_id)
                SELECT id, kind, parent_id, ordinal, depth, page_start, page_end, title FROM up ORDER BY depth""",
                (doc_id, node_id, doc_id)).fetchall()
            return [self._row(r) for r in rows]

    def section_at_page(self, doc_id: str, page: int) -> Optional[NodeRow]:
        """The deepest, latest section that starts on or before ``page`` and reaches it."""
        with self._lock:
            r = self._db.execute(
                f"SELECT {self._COLS} FROM nodes WHERE doc_id=? AND kind='section' AND page_start<=? AND page_end>=? "
                "ORDER BY depth DESC, ordinal DESC LIMIT 1", (doc_id, page, page)).fetchone()
            if r is None:
                r = self._db.execute(f"SELECT {self._COLS} FROM nodes WHERE doc_id=? AND kind='section' AND page_start<=? "
                                     "ORDER BY ordinal DESC LIMIT 1", (doc_id, page)).fetchone()
            return self._row(r) if r else None

    def edges(self, doc_id: str, kind: Optional[EdgeKind] = None) -> List[tuple]:
        with self._lock:
            sql = "SELECT src, dst, kind FROM edges WHERE doc_id=?" + (" AND kind=?" if kind else "")
            return list(self._db.execute(sql, (doc_id, kind.value) if kind else (doc_id,)))

    def all_text_cells(self) -> List[str]:
        """Every text value stored (used by tests to prove no body text is kept here)."""
        with self._lock:
            return [str(v) for table in ("documents", "nodes", "edges")
                    for row in self._db.execute(f"SELECT * FROM {table}") for v in row if isinstance(v, str)]


class JsonContentStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self._dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id) + ".json")

    def save(self, structure: DocumentStructure) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(structure.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, self._path(structure.doc_id))

    def load(self, doc_id: str) -> DocumentStructure:
        with open(self._path(doc_id), encoding="utf-8") as f:
            return DocumentStructure.from_dict(json.load(f))

    def exists(self, doc_id: str) -> bool:
        return self._path(doc_id).exists()

    def delete(self, doc_id: str) -> None:
        self._path(doc_id).unlink(missing_ok=True)
