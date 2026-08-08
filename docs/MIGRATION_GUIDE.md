# Database Migration Guide

## Overview
This guide explains how to work with database migrations in the Doc Explainer project.

## Quick Start

### Initialize Database
```python
from src.database import get_db_manager

# Initialize database and run all pending migrations
db_manager = get_db_manager()
```

The database will be created at `data/db.sqlite` with all required tables.

### Execute Queries
```python
db_manager = get_db_manager()

# Fetch a single row
row = db_manager.fetch_one(
    "SELECT * FROM concept_mastery WHERE user_id = ?",
    ("user123",)
)

# Fetch all rows
rows = db_manager.fetch_all(
    "SELECT * FROM quiz_responses WHERE user_id = ? ORDER BY timestamp DESC",
    ("user123",)
)

# Execute insert/update/delete
db_manager.execute_query(
    "INSERT INTO concept_mastery (user_id, concept_id, mastery_level) VALUES (?, ?, ?)",
    ("user123", "algebra", 0.85)
)
```

## Creating New Migrations

### Step 1: Create Migration File
Create a new SQL file in `src/store/migrations/`:
```
002_add_your_feature.sql
```

File naming convention: `NNN_description_in_snake_case.sql`
- Use zero-padded numbers (001, 002, etc.)
- Use lowercase with underscores

### Step 2: Write Migration SQL
```sql
-- Migration: Description of changes
-- Purpose: Why these changes are needed

CREATE TABLE your_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    -- ... columns
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_your_table_user_id ON your_table(user_id);
```

### Step 3: Test Migration
```python
import sqlite3
from pathlib import Path

# Test in-memory
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

migration_sql = Path('src/store/migrations/002_add_your_feature.sql').read_text()
cursor.executescript(migration_sql)

# Verify tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables created: {tables}")
```

### Step 4: Deploy
Migrations are applied automatically on next database initialization:
```python
from src.database import get_db_manager
db_manager = get_db_manager()  # Runs all pending migrations
```

## Migration Tracking

Migrations are tracked in the `_migrations` table:
```sql
SELECT name, applied_at FROM _migrations ORDER BY applied_at;
```

Once applied, a migration will never be run again (idempotent tracking).

## Best Practices

### 1. Idempotency
Always use `CREATE TABLE IF NOT EXISTS`:
```sql
CREATE TABLE IF NOT EXISTS new_table (
    id INTEGER PRIMARY KEY
);
```

### 2. Avoid Breaking Changes
- Don't drop columns in existing tables without adding backwards compatibility
- Don't change column types mid-project
- Always use `ON DELETE CASCADE` for foreign keys to prevent orphaned data

### 3. Include Comments
```sql
-- Migration: Add user preferences
-- Purpose: Support feature flag system for A/B testing
-- Affected tables: users
-- Rollback: DROP TABLE user_preferences;
```

### 4. Test Thoroughly
- Test migration in isolation
- Test with existing data
- Verify performance impact on large datasets

### 5. Use Proper Constraints
```sql
CREATE TABLE example (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active')),
    confidence REAL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    UNIQUE(user_id, concept_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
```

### 6. Add Indexes for Performance
```sql
-- Index frequently-queried columns
CREATE INDEX idx_table_user_id ON table(user_id);
CREATE INDEX idx_table_timestamp ON table(timestamp DESC);

-- Composite index for multi-column queries
CREATE INDEX idx_table_user_date ON table(user_id, timestamp);
```

## Troubleshooting

### Migration Failed
Check the error log and the migration SQL:
```bash
# View migration SQL
cat src/store/migrations/001_add_advanced_features.sql
```

### Database Locked
Stop any other processes using the database:
```bash
# Find processes
lsof data/db.sqlite
```

### Foreign Key Constraint Errors
Ensure foreign keys are properly configured:
```sql
PRAGMA foreign_keys=ON;
SELECT * FROM sqlite_master WHERE type='table' AND sql LIKE '%FOREIGN KEY%';
```

### Verify Schema
```sql
-- List all tables
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

-- Show table structure
PRAGMA table_info(table_name);

-- Show indexes
PRAGMA index_list(table_name);
```

## Performance Considerations

### Large Tables
For migrations adding indexes to existing large tables:
```sql
-- Analyze table statistics
ANALYZE table_name;

-- Check index effectiveness
PRAGMA index_info(index_name);
```

### Database Maintenance
```sql
-- Optimize database file
VACUUM;

-- Update statistics
ANALYZE;

-- Check database integrity
PRAGMA integrity_check;
```

## Integration with Application

### Initialization Point
Call database initialization at application startup:
```python
from src.database import get_db_manager

# In your main entry point
def main():
    db_manager = get_db_manager()
    # Rest of application code
```

### Connection Pooling
For high-concurrency applications, consider wrapping:
```python
from src.store.db import DatabaseManager

db_manager = DatabaseManager()
db_manager.initialize()

# Use for all database operations
row = db_manager.fetch_one(query, params)
```

## Documentation

See `docs/DATABASE_SCHEMA.md` for complete schema documentation including:
- Table structures
- Indexes
- Constraints
- Usage examples
