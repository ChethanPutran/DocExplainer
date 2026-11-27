"""SQLite store for the temporal knowledge graph.

Every row carries the ordinal/step at which it first appeared, so "the graph as of position N" is a
plain filter and no replay is needed. The ``deltas`` table additionally keeps each step's delta as JSON:
the append-only history, from which the whole graph can be rebuilt or audited.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..common.knowledge import (Concept, KnowledgeDelta, Mention, MentionRole, Relation, RelationType, StepRecord)
from .canonical import normalize_name

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (doc_id TEXT, key TEXT, value TEXT, PRIMARY KEY (doc_id, key));
CREATE TABLE IF NOT EXISTS concepts (
    doc_id TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL, norm TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
    introduced_step INTEGER NOT NULL, introduced_ordinal INTEGER NOT NULL, first_node_id TEXT NOT NULL DEFAULT '',
    definition TEXT NOT NULL DEFAULT '', definition_node_id TEXT NOT NULL DEFAULT '', defined_ordinal INTEGER,
    importance INTEGER NOT NULL DEFAULT 0, mention_count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (doc_id, id));
CREATE INDEX IF NOT EXISTS concepts_ord ON concepts (doc_id, introduced_ordinal);
CREATE TABLE IF NOT EXISTS definitions (doc_id TEXT, concept_id TEXT, ordinal INTEGER, text TEXT, node_id TEXT);
CREATE INDEX IF NOT EXISTS definitions_c ON definitions (doc_id, concept_id, ordinal);
CREATE TABLE IF NOT EXISTS mentions (doc_id TEXT, concept_id TEXT, node_id TEXT, ordinal INTEGER, step INTEGER, role TEXT,
    PRIMARY KEY (doc_id, concept_id, node_id));
CREATE INDEX IF NOT EXISTS mentions_ord ON mentions (doc_id, ordinal);
CREATE TABLE IF NOT EXISTS relations (doc_id TEXT, src TEXT, dst TEXT, type TEXT, first_ordinal INTEGER, first_step INTEGER,
    strength INTEGER, evidence TEXT, PRIMARY KEY (doc_id, src, dst, type));
CREATE TABLE IF NOT EXISTS steps (doc_id TEXT, step INTEGER, section_id TEXT, ordinal_start INTEGER, ordinal_end INTEGER,
    words INTEGER, status TEXT, new_concepts INTEGER, new_relations INTEGER, mentions INTEGER, known_mentions INTEGER,
    error TEXT, PRIMARY KEY (doc_id, step));
CREATE TABLE IF NOT EXISTS deltas (doc_id TEXT, step INTEGER, payload TEXT, PRIMARY KEY (doc_id, step));
"""
_TABLES = ("meta", "concepts", "definitions", "mentions", "relations", "steps", "deltas")
_CONCEPT_COLS = ("id, name, aliases, introduced_step, introduced_ordinal, first_node_id, definition, "
                 "definition_node_id, defined_ordinal, importance, mention_count")


def _concept(r) -> Concept:
    return Concept(id=r[0], name=r[1], aliases=json.loads(r[2]), introduced_step=r[3], introduced_ordinal=r[4],
                   first_node_id=r[5], definition=r[6], definition_node_id=r[7], defined_ordinal=r[8],
                   importance=r[9], mention_count=r[10])


def _relation(r) -> Relation:
    return Relation(src=r[0], dst=r[1], type=RelationType(r[2]), first_ordinal=r[3], first_step=r[4],
                    strength=r[5], evidence=json.loads(r[6]))


def delta_to_dict(d: KnowledgeDelta) -> dict:
    """JSON-safe form of a delta (enums as values)."""
    out = asdict(d)
    for m in out["mentions"]:
        m["role"] = m["role"].value if hasattr(m["role"], "value") else m["role"]
    for r in out["new_relations"]:
        r["type"] = r["type"].value if hasattr(r["type"], "value") else r["type"]
    return out


def delta_from_dict(d: dict) -> KnowledgeDelta:
    """Inverse of ``delta_to_dict``."""
    return KnowledgeDelta(
        step=d["step"], section_id=d["section_id"], ordinal_start=d["ordinal_start"], ordinal_end=d["ordinal_end"],
        new_concepts=[Concept(**c) for c in d["new_concepts"]],
        updated_aliases=[(a, list(b)) for a, b in d["updated_aliases"]],
        definitions=[tuple(x) for x in d["definitions"]],
        mentions=[Mention(m["concept_id"], m["node_id"], m["ordinal"], m["step"], MentionRole(m["role"])) for m in d["mentions"]],
        new_relations=[Relation(r["src"], r["dst"], RelationType(r["type"]), r["first_ordinal"], r["first_step"],
                                r["strength"], list(r["evidence"])) for r in d["new_relations"]],
        strengthened=[tuple(x) for x in d["strengthened"]], rejected=list(d["rejected"]),
        importance=[tuple(x) for x in d["importance"]], words=d["words"])


class SqliteKnowledgeStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ writes
    def apply(self, doc_id: str, delta: KnowledgeDelta, record: StepRecord) -> None:
        """Persist one step atomically: the state tables, the step record and the delta history."""
        with self._lock, self._db:
            for c in delta.new_concepts:
                self._db.execute(
                    "INSERT INTO concepts (doc_id, id, name, norm, aliases, introduced_step, introduced_ordinal, first_node_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (doc_id, c.id, c.name, normalize_name(c.name), json.dumps(c.aliases), c.introduced_step,
                     c.introduced_ordinal, c.first_node_id))
            for cid, aliases in delta.updated_aliases:
                row = self._db.execute("SELECT aliases FROM concepts WHERE doc_id=? AND id=?", (doc_id, cid)).fetchone()
                merged = list(dict.fromkeys(json.loads(row[0]) + aliases))
                self._db.execute("UPDATE concepts SET aliases=? WHERE doc_id=? AND id=?", (json.dumps(merged), doc_id, cid))
            for cid, ordinal, text, node_id in delta.definitions:
                self._db.execute("INSERT INTO definitions VALUES (?,?,?,?,?)", (doc_id, cid, ordinal, text, node_id))
                self._db.execute(
                    "UPDATE concepts SET definition=?, definition_node_id=?, defined_ordinal=? "
                    "WHERE doc_id=? AND id=? AND defined_ordinal IS NULL", (text, node_id, ordinal, doc_id, cid))
            for m in delta.mentions:
                self._db.execute("INSERT OR IGNORE INTO mentions VALUES (?,?,?,?,?,?)",
                                 (doc_id, m.concept_id, m.node_id, m.ordinal, m.step, m.role.value))
                self._db.execute("UPDATE concepts SET mention_count = mention_count + 1 WHERE doc_id=? AND id=?",
                                 (doc_id, m.concept_id))
            for cid, weight in delta.importance:
                self._db.execute("UPDATE concepts SET importance = importance + ? WHERE doc_id=? AND id=?", (weight, doc_id, cid))
            for r in delta.new_relations:
                self._db.execute("INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?,?,?,?)",
                                 (doc_id, r.src, r.dst, r.type.value, r.first_ordinal, r.first_step, r.strength,
                                  json.dumps(r.evidence)))
            for src, dst, kind, node_id in delta.strengthened:
                row = self._db.execute("SELECT evidence FROM relations WHERE doc_id=? AND src=? AND dst=? AND type=?",
                                       (doc_id, src, dst, kind)).fetchone()
                evidence = json.loads(row[0])
                if node_id and node_id not in evidence and len(evidence) < 5:
                    evidence.append(node_id)
                self._db.execute("UPDATE relations SET strength = strength + 1, evidence=? WHERE doc_id=? AND src=? AND dst=? AND type=?",
                                 (json.dumps(evidence), doc_id, src, dst, kind))
            self._write_step(doc_id, record)
            self._db.execute("INSERT OR REPLACE INTO deltas VALUES (?,?,?)",
                             (doc_id, delta.step, json.dumps(delta_to_dict(delta))))

    def record_failure(self, doc_id: str, record: StepRecord) -> None:
        with self._lock, self._db:
            self._write_step(doc_id, record)

    def _write_step(self, doc_id: str, r: StepRecord) -> None:
        self._db.execute("INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (doc_id, r.step, r.section_id, r.ordinal_start, r.ordinal_end, r.words, r.status,
                          r.new_concepts, r.new_relations, r.mentions, r.known_mentions, r.error))

    def reset(self, doc_id: str) -> None:
        with self._lock, self._db:
            for table in _TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))

    def rewind(self, doc_id: str, to_step: int) -> None:
        """Make the graph exactly what it was before step ``to_step`` ran, by rebuilding it from the delta history.

        Needed when a step must be redone after later ones finished: knowledge introduced "later" must not
        remain in the graph, or it would pre-date what the redone step introduces."""
        with self._lock:
            meta = {r[0]: r[1] for r in self._db.execute("SELECT key, value FROM meta WHERE doc_id=?", (doc_id,))}
            records = {r.step: r for r in self.steps(doc_id) if r.step < to_step and r.status == "done"}
            deltas = [delta_from_dict(d) for d in self.deltas(doc_id) if d["step"] < to_step and d["step"] in records]
            self.reset(doc_id)
            for k, v in meta.items():
                self.set_meta(doc_id, k, v)
            for d in deltas:
                self.apply(doc_id, d, records[d.step])

    def set_meta(self, doc_id: str, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?)", (doc_id, key, value))

    def get_meta(self, doc_id: str, key: str) -> Optional[str]:
        with self._lock:
            r = self._db.execute("SELECT value FROM meta WHERE doc_id=? AND key=?", (doc_id, key)).fetchone()
            return r[0] if r else None

    # ------------------------------------------------------------------ reads
    def concepts(self, doc_id: str, upto_ordinal: Optional[int] = None) -> List[Concept]:
        with self._lock:
            sql = f"SELECT {_CONCEPT_COLS} FROM concepts WHERE doc_id=?"
            args: list = [doc_id]
            if upto_ordinal is not None:
                sql += " AND introduced_ordinal<=?"
                args.append(upto_ordinal)
            return [_concept(r) for r in self._db.execute(sql + " ORDER BY introduced_ordinal, id", args)]

    def concept(self, doc_id: str, concept_id: str) -> Optional[Concept]:
        with self._lock:
            r = self._db.execute(f"SELECT {_CONCEPT_COLS} FROM concepts WHERE doc_id=? AND id=?", (doc_id, concept_id)).fetchone()
            return _concept(r) if r else None

    def next_concept_number(self, doc_id: str) -> int:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM concepts WHERE doc_id=?", (doc_id,))]
            return 1 + max([int(i[1:]) for i in ids if i[1:].isdigit()] or [0])

    def relations(self, doc_id: str, upto_ordinal: Optional[int] = None, type: Optional[RelationType] = None) -> List[Relation]:
        with self._lock:
            sql = "SELECT src, dst, type, first_ordinal, first_step, strength, evidence FROM relations WHERE doc_id=?"
            args: list = [doc_id]
            if upto_ordinal is not None:
                sql += " AND first_ordinal<=?"
                args.append(upto_ordinal)
            if type is not None:
                sql += " AND type=?"
                args.append(type.value)
            return [_relation(r) for r in self._db.execute(sql + " ORDER BY first_ordinal, src, dst", args)]

    def mentions(self, doc_id: str, concept_id: Optional[str] = None, lo: Optional[int] = None,
                 hi: Optional[int] = None) -> List[Mention]:
        with self._lock:
            sql = "SELECT concept_id, node_id, ordinal, step, role FROM mentions WHERE doc_id=?"
            args: list = [doc_id]
            for cond, val in (("concept_id=?", concept_id), ("ordinal>=?", lo), ("ordinal<=?", hi)):
                if val is not None:
                    sql += f" AND {cond}"
                    args.append(val)
            return [Mention(r[0], r[1], r[2], r[3], MentionRole(r[4])) for r in self._db.execute(sql + " ORDER BY ordinal", args)]

    def definitions(self, doc_id: str, concept_id: Optional[str] = None) -> List[tuple]:
        """(concept id, ordinal, text, node id), in reading order."""
        with self._lock:
            sql = "SELECT concept_id, ordinal, text, node_id FROM definitions WHERE doc_id=?"
            args: list = [doc_id]
            if concept_id:
                sql += " AND concept_id=?"
                args.append(concept_id)
            return list(self._db.execute(sql + " ORDER BY ordinal", args))

    def steps(self, doc_id: str) -> List[StepRecord]:
        with self._lock:
            return [StepRecord(step=r[0], section_id=r[1], ordinal_start=r[2], ordinal_end=r[3], words=r[4], status=r[5],
                               new_concepts=r[6], new_relations=r[7], mentions=r[8], known_mentions=r[9], error=r[10])
                    for r in self._db.execute(
                        "SELECT step, section_id, ordinal_start, ordinal_end, words, status, new_concepts, new_relations, "
                        "mentions, known_mentions, error FROM steps WHERE doc_id=? ORDER BY step", (doc_id,))]

    def deltas(self, doc_id: str) -> List[dict]:
        with self._lock:
            return [json.loads(r[0]) for r in self._db.execute("SELECT payload FROM deltas WHERE doc_id=? ORDER BY step", (doc_id,))]

    def has_knowledge(self, doc_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM steps WHERE doc_id=? AND status='done' LIMIT 1", (doc_id,)).fetchone() is not None
