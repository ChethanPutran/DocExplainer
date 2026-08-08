# Database Schema Documentation

## Overview
This document describes the database schema for the advanced learning features in the Doc Explainer system. All tables are created using SQLite and are optimized for common query patterns.

## Tables

### 1. `concept_mastery`
Tracks user mastery levels for individual concepts.

**Purpose**: Store learned mastery levels to guide personalized learning paths and adaptive difficulty.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `user_id` (TEXT, FOREIGN KEY): Reference to user
- `concept_id` (TEXT): Identifier for the concept being mastered
- `mastery_level` (REAL 0-1): Current mastery score (0.0 = none, 1.0 = expert)
- `confidence` (REAL 0-1): Confidence in the mastery measurement
- `last_updated` (TIMESTAMP): When the mastery level was last updated
- `created_at` (TIMESTAMP): When the record was created

**Constraints**:
- Unique constraint on (user_id, concept_id)
- Foreign key to users table (cascade delete)
- CHECK constraints on 0-1 range for mastery_level and confidence

**Indexes**:
- user_id, concept_id, mastery_level, (user_id, concept_id)

---

### 2. `quiz_responses`
Records individual quiz responses for analysis and adaptive learning.

**Purpose**: Enable detailed tracking of quiz performance for analytics and difficulty adjustment.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `user_id` (TEXT, FOREIGN KEY): Reference to user
- `quiz_id` (TEXT): Identifier for the quiz
- `question_id` (TEXT): Identifier for the question
- `response` (TEXT): User's response or answer
- `is_correct` (BOOLEAN): Whether the response was correct
- `difficulty_level` (REAL 0-1): Difficulty rating of the question
- `timestamp` (TIMESTAMP): When the response was submitted

**Constraints**:
- Foreign key to users table (cascade delete)
- CHECK constraint on 0-1 range for difficulty_level

**Indexes**:
- user_id, quiz_id, question_id, is_correct, timestamp, (user_id, quiz_id)

---

### 3. `document_concepts`
Maps concepts to documents and paragraphs with confidence scoring.

**Purpose**: Tag documents/paragraphs with concepts for concept-based retrieval and analysis.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `doc_id` (TEXT): Document identifier
- `concept_id` (TEXT): Concept identifier
- `paragraph_id` (TEXT): Optional paragraph-level identifier
- `confidence_score` (REAL 0-1): Confidence that this concept appears in the document/paragraph
- `tagged_by` (TEXT): 'auto' (system-generated) or 'manual' (human-tagged)
- `created_at` (TIMESTAMP): When the tag was created
- `updated_at` (TIMESTAMP): When the tag was last updated

**Constraints**:
- Unique constraint on (doc_id, concept_id, paragraph_id)
- CHECK constraint on 0-1 range for confidence_score
- CHECK constraint on 'auto'/'manual' for tagged_by

**Indexes**:
- doc_id, concept_id, paragraph_id, confidence_score, tagged_by, (doc_id, concept_id)

---

### 4. `explanation_feedback`
Collects user feedback on explanations for quality improvement.

**Purpose**: Track explanation quality and collect user satisfaction metrics.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `user_id` (TEXT, FOREIGN KEY): Reference to user providing feedback
- `explanation_id` (TEXT): Identifier of the explanation being rated
- `rating` (INTEGER 1-5): User's rating (1 = poor, 5 = excellent)
- `feedback_text` (TEXT): Optional detailed feedback
- `timestamp` (TIMESTAMP): When feedback was submitted

**Constraints**:
- Foreign key to users table (cascade delete)
- CHECK constraint on 1-5 range for rating

**Indexes**:
- user_id, explanation_id, rating, timestamp, (user_id, explanation_id)

---

### 5. `learning_paths`
Defines personalized learning sequences for users.

**Purpose**: Store learning pathways with progress tracking for adaptive curriculum.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `path_id` (TEXT): Learning path identifier
- `user_id` (TEXT, FOREIGN KEY): Reference to user
- `concept_id` (TEXT): Concept in the learning sequence
- `sequence_order` (INTEGER): Position in the learning path sequence
- `status` (TEXT): 'pending', 'in_progress', 'completed', or 'skipped'
- `created_at` (TIMESTAMP): When the learning path was created
- `updated_at` (TIMESTAMP): When the status was last updated
- `completed_at` (TIMESTAMP): When the concept was completed (NULL if not completed)

**Constraints**:
- Unique constraint on (path_id, user_id, concept_id)
- Foreign key to users table (cascade delete)
- CHECK constraint on valid status values

**Indexes**:
- path_id, user_id, concept_id, status, sequence_order, (user_id, path_id)

---

### 6. `concept_transfer`
Tracks transfer of learning between related concepts/documents.

**Purpose**: Analyze how learning in one domain transfers to another (e.g., using math concepts from Doc A to solve problems in Doc B).

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing unique identifier
- `source_doc` (TEXT): Document where learning originated
- `target_doc` (TEXT): Document where learning transferred to
- `source_concept` (TEXT): Original concept learned
- `target_concept` (TEXT): Related concept where transfer occurred
- `transfer_score` (REAL 0-1): Strength of transfer (0.0 = none, 1.0 = perfect transfer)
- `transfer_count` (INTEGER): Number of times transfer was detected
- `last_detected` (TIMESTAMP): When transfer was last observed
- `created_at` (TIMESTAMP): When the relationship was first recorded

**Constraints**:
- Unique constraint on (source_doc, target_doc, source_concept, target_concept)
- CHECK constraint on 0-1 range for transfer_score

**Indexes**:
- source_doc, target_doc, source_concept, target_concept, transfer_score, (source_doc, target_doc), (source_concept, target_concept)

---

### 7. `users`
Base users table for foreign key references.

**Purpose**: Central user registry for foreign key relationships.

**Columns**:
- `user_id` (TEXT PRIMARY KEY): Unique user identifier
- `created_at` (TIMESTAMP): When the user record was created
- `updated_at` (TIMESTAMP): When the user record was last updated

**Indexes**:
- created_at

---

## Migration System

### `_migrations` Table
Tracks which migrations have been applied to the database.

**Columns**:
- `id` (INTEGER PRIMARY KEY): Auto-incrementing identifier
- `name` (TEXT UNIQUE): Migration filename
- `applied_at` (TIMESTAMP): When the migration was applied

### Migration Files

**Location**: `src/store/migrations/`

**Format**: SQL files named with pattern `NNN_description.sql` (e.g., `001_add_advanced_features.sql`)

**Execution**: Migrations are applied automatically when the database is initialized.

---

## Performance Considerations

### Indexes
All frequently-queried columns have dedicated indexes:
- Single-column indexes for exact matches and range queries
- Composite indexes for multi-column WHERE/JOIN conditions
- 41 total indexes across all tables

### Foreign Keys
- Foreign keys are configured with CASCADE DELETE to maintain referential integrity
- Foreign key constraints must be explicitly enabled with `PRAGMA foreign_keys=ON`

### Data Types
- REAL fields are used for confidence/mastery scores (0-1 range)
- INTEGER used for counts and sequence ordering
- TEXT used for identifiers to allow flexible ID schemes
- TIMESTAMP for audit trails and time-series analysis

---

## Usage Examples

### Query mastery levels for a user
```sql
SELECT concept_id, mastery_level, confidence, last_updated
FROM concept_mastery
WHERE user_id = 'user123'
ORDER BY mastery_level DESC;
```

### Find documents where a concept appears with high confidence
```sql
SELECT DISTINCT doc_id, paragraph_id, confidence_score
FROM document_concepts
WHERE concept_id = 'algebra'
AND confidence_score > 0.7
ORDER BY confidence_score DESC;
```

### Get learning path progress
```sql
SELECT concept_id, sequence_order, status
FROM learning_paths
WHERE path_id = 'path_001' AND user_id = 'user123'
ORDER BY sequence_order;
```

### Analyze concept transfer between documents
```sql
SELECT source_concept, target_concept, transfer_score, transfer_count
FROM concept_transfer
WHERE source_doc = 'doc_001'
AND transfer_score > 0.5
ORDER BY transfer_score DESC;
```

---

## Maintenance

### Backing Up
All data can be backed up using standard SQLite tools:
```bash
sqlite3 data/db.sqlite ".backup 'backup.sqlite'"
```

### Analyzing Performance
```sql
-- Check table sizes
SELECT name, page_count * page_size as bytes 
FROM pragma_page_count(), pragma_page_size(), sqlite_master
WHERE type='table';

-- Check index usage
PRAGMA index_list(table_name);
```

### Vacuuming
```sql
-- Optimize database file
VACUUM;
ANALYZE;
```
