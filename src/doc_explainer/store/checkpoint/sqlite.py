import sqlite3
import os
from typing import Optional
from .base import CheckpointStore

class SQLiteCheckpointStore(CheckpointStore):
    def __init__(self, db_path: str = "./db/checkpoints.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                document_id TEXT,
                section_id TEXT,
                status TEXT,
                PRIMARY KEY (document_id, section_id)
            )
        ''')
        conn.commit()
        conn.close()

    def mark_started(self, document_id: str, section_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO checkpoints (document_id, section_id, status) VALUES (?, ?, 'started')",
                  (document_id, section_id))
        conn.commit()
        conn.close()

    def mark_completed(self, document_id: str, section_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE checkpoints SET status = 'completed' WHERE document_id = ? AND section_id = ?",
                  (document_id, section_id))
        conn.commit()
        conn.close()

    def is_completed(self, document_id: str, section_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT status FROM checkpoints WHERE document_id = ? AND section_id = ?", (document_id, section_id))
        row = c.fetchone()
        conn.close()
        return row is not None and row[0] == 'completed'

    def get_last_completed(self, document_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT section_id FROM checkpoints WHERE document_id = ? AND status = 'completed' ORDER BY rowid DESC LIMIT 1",
                  (document_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None