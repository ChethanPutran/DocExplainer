"""Database initialization and connection management."""

import sqlite3
from pathlib import Path
from typing import Optional


class DatabaseManager:
    """Manages SQLite database connections and migrations."""
    
    def __init__(self, db_path: str = "data/db.sqlite"):
        self.db_path = db_path
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize(self):
        """Initialize database and run migrations."""
        from .migrations import run_migrations
        run_migrations(self.db_path)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        conn.close()
        return cursor
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Fetch a single row."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list:
        """Fetch all rows."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# Global database instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()
    return _db_manager
