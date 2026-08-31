import sqlite3
import os
from typing import Optional

from doc_explainer.core.document.parser import ParserFactory
from doc_explainer.core.document.processor.hierarchy import HierarchicalProcessor
from doc_explainer.core.document.engine import DocumentEngine
from doc_explainer.store.document.repository import DocumentRepository
from doc_explainer.store.checkpoint.base import CheckpointStore
from doc_explainer.core.document.manager import DocumentManager
from doc_explainer.orchestrator.artifacts.store import ArtifactStore


class ServiceRegistry:
    """Global registry for shared services."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, **kwargs):
        if self._initialized:
            return
        self.parser_factory: ParserFactory = kwargs.get("parser_factory", ParserFactory())
        self.processor: HierarchicalProcessor = kwargs.get("processor",HierarchicalProcessor()) 
        self.engine: DocumentEngine = kwargs.get("engine", DocumentEngine())       
        self.repository: DocumentRepository = kwargs.get("repository", DocumentRepository())
        self.checkpoint_store: CheckpointStore = kwargs.get("checkpoint_store", CheckpointStore())
        self.artifact_store: ArtifactStore = kwargs.get("artifact_store", ArtifactStore())
        self.manager = DocumentManager(
            repository=self.repository,
            document_engine=self.engine
        )
        self._initialized = True



class DocumentRegistry:
    """Tracks which documents have been processed and their run IDs."""
    def __init__(self, db_path: str = "./db/registry.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS processed_docs (
                file_path TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def mark_processed(self, file_path: str, doc_id: str, run_id: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO processed_docs (file_path, doc_id, run_id) VALUES (?, ?, ?)",
            (file_path, doc_id, run_id)
        )
        conn.commit()
        conn.close()

    def get_processed(self, file_path: str) -> Optional[tuple]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT doc_id, run_id FROM processed_docs WHERE file_path = ?", (file_path,))
        row = c.fetchone()
        conn.close()
        return row if row else None

registry = ServiceRegistry()