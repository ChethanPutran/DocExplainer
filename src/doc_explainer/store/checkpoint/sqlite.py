from __future__ import annotations

import os
import sqlite3
from typing import Optional

from .base import CheckpointStore


class SQLiteCheckpointStore(CheckpointStore):
    """SQLite-backed document and section processing checkpoints."""

    def __init__(self, db_path: str = "./db/checkpoints.db") -> None:
        self.db_path = db_path
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_runs (
                    namespace TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    namespace TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    PRIMARY KEY (namespace, section_id),
                    FOREIGN KEY (namespace)
                        REFERENCES processing_runs(namespace)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_namespace_status
                ON checkpoints (namespace, status)
                """
            )

    def start(self, namespace: str, file_path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO processing_runs
                    (namespace, file_path, status, started_at, completed_at)
                VALUES (?, ?, 'started', CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(namespace) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'started',
                    started_at = CURRENT_TIMESTAMP,
                    completed_at = NULL
                """,
                (namespace, file_path),
            )

    def complete(self, namespace: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE processing_runs
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE namespace = ?
                """,
                (namespace,),
            )

    def mark_started(self, namespace: str, section_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints
                    (namespace, section_id, status, error, started_at, completed_at)
                VALUES (?, ?, 'started', NULL, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(namespace, section_id) DO UPDATE SET
                    status = 'started',
                    error = NULL,
                    started_at = CURRENT_TIMESTAMP,
                    completed_at = NULL
                """,
                (namespace, section_id),
            )

    def mark_completed(self, namespace: str, section_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints
                    (namespace, section_id, status, error, started_at, completed_at)
                VALUES (?, ?, 'completed', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(namespace, section_id) DO UPDATE SET
                    status = 'completed',
                    error = NULL,
                    completed_at = CURRENT_TIMESTAMP
                """,
                (namespace, section_id),
            )

    def mark_section_failed(
        self,
        namespace: str,
        section_id: str,
        error: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints
                    (namespace, section_id, status, error, started_at, completed_at)
                VALUES (?, ?, 'failed', ?, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(namespace, section_id) DO UPDATE SET
                    status = 'failed',
                    error = excluded.error,
                    completed_at = NULL
                """,
                (namespace, section_id, error),
            )

    def is_completed(self, namespace: str, section_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM checkpoints
                WHERE namespace = ? AND section_id = ? AND status = 'completed'
                LIMIT 1
                """,
                (namespace, section_id),
            ).fetchone()
        return row is not None

    def get_last_completed(self, namespace: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT section_id FROM checkpoints
                WHERE namespace = ? AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (namespace,),
            ).fetchone()
        return row["section_id"] if row is not None else None

    def is_run_complete(self, namespace: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM processing_runs
                WHERE namespace = ? AND status = 'completed'
                LIMIT 1
                """,
                (namespace,),
            ).fetchone()
        return row is not None

    def mark_registration_complete(self, namespace: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE processing_runs
                SET status = 'registration_completed',
                    completed_at = CURRENT_TIMESTAMP
                WHERE namespace = ?
                """,
                (namespace,),
            )

    def is_registration_complete(self, namespace: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM processing_runs
                WHERE namespace = ? AND status = 'registration_completed'
                LIMIT 1
                """,
                (namespace,),
            ).fetchone()
        return row is not None
