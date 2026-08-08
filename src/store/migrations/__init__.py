"""Database migrations module for doc_explainer."""

__all__ = ["run_migrations"]


def run_migrations(db_path: str) -> None:
    """Run all pending migrations."""
    import sqlite3
    from pathlib import Path
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create migrations tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Get all migration files
    migrations_dir = Path(__file__).parent
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    for migration_file in migration_files:
        migration_name = migration_file.name
        
        # Check if already applied
        cursor.execute("SELECT 1 FROM _migrations WHERE name = ?", (migration_name,))
        if cursor.fetchone():
            continue
        
        # Read and execute migration
        sql_content = migration_file.read_text()
        cursor.executescript(sql_content)
        
        # Record migration
        cursor.execute("INSERT INTO _migrations (name) VALUES (?)", (migration_name,))
        conn.commit()
        print(f"Applied migration: {migration_name}")
    
    conn.close()
