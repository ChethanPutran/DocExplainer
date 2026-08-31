-- Migration: Add advanced learning features tables
-- Description: Create tables for concept mastery tracking, quiz responses, document concepts, 
--              explanation feedback, learning paths, and concept transfer analysis

-- Table: concept_mastery
-- Tracks user mastery levels for individual concepts
CREATE TABLE IF NOT EXISTS concept_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    mastery_level REAL NOT NULL DEFAULT 0.0 CHECK (mastery_level >= 0.0 AND mastery_level <= 1.0),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, concept_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_concept_mastery_user_id ON concept_mastery(user_id);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_concept_id ON concept_mastery(concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_mastery_level ON concept_mastery(mastery_level);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_user_concept ON concept_mastery(user_id, concept_id);

-- Table: quiz_responses
-- Records individual quiz responses for analysis and adaptive learning
CREATE TABLE IF NOT EXISTS quiz_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    quiz_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    response TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    difficulty_level REAL DEFAULT 0.5 CHECK (difficulty_level >= 0.0 AND difficulty_level <= 1.0),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_quiz_responses_user_id ON quiz_responses(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_quiz_id ON quiz_responses(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_question_id ON quiz_responses(question_id);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_is_correct ON quiz_responses(is_correct);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_timestamp ON quiz_responses(timestamp);
CREATE INDEX IF NOT EXISTS idx_quiz_responses_user_quiz ON quiz_responses(user_id, quiz_id);

-- Table: document_concepts
-- Maps concepts to documents and paragraphs with confidence scoring
CREATE TABLE IF NOT EXISTS document_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    paragraph_id TEXT,
    confidence_score REAL NOT NULL DEFAULT 0.5 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    tagged_by TEXT NOT NULL DEFAULT 'auto' CHECK (tagged_by IN ('auto', 'manual')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doc_id, concept_id, paragraph_id)
);

CREATE INDEX IF NOT EXISTS idx_document_concepts_doc_id ON document_concepts(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_concepts_concept_id ON document_concepts(concept_id);
CREATE INDEX IF NOT EXISTS idx_document_concepts_paragraph_id ON document_concepts(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_document_concepts_confidence_score ON document_concepts(confidence_score);
CREATE INDEX IF NOT EXISTS idx_document_concepts_tagged_by ON document_concepts(tagged_by);
CREATE INDEX IF NOT EXISTS idx_document_concepts_doc_concept ON document_concepts(doc_id, concept_id);

-- Table: explanation_feedback
-- Collects user feedback on explanations for quality improvement
CREATE TABLE IF NOT EXISTS explanation_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    explanation_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_explanation_feedback_user_id ON explanation_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_explanation_feedback_explanation_id ON explanation_feedback(explanation_id);
CREATE INDEX IF NOT EXISTS idx_explanation_feedback_rating ON explanation_feedback(rating);
CREATE INDEX IF NOT EXISTS idx_explanation_feedback_timestamp ON explanation_feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_explanation_feedback_user_explanation ON explanation_feedback(user_id, explanation_id);

-- Table: learning_paths
-- Defines personalized learning sequences for users
CREATE TABLE IF NOT EXISTS learning_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    sequence_order INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(path_id, user_id, concept_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_learning_paths_path_id ON learning_paths(path_id);
CREATE INDEX IF NOT EXISTS idx_learning_paths_user_id ON learning_paths(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_paths_concept_id ON learning_paths(concept_id);
CREATE INDEX IF NOT EXISTS idx_learning_paths_status ON learning_paths(status);
CREATE INDEX IF NOT EXISTS idx_learning_paths_sequence_order ON learning_paths(sequence_order);
CREATE INDEX IF NOT EXISTS idx_learning_paths_user_path ON learning_paths(user_id, path_id);

-- Table: concept_transfer
-- Tracks transfer of learning between related concepts/documents
CREATE TABLE IF NOT EXISTS concept_transfer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc TEXT NOT NULL,
    target_doc TEXT NOT NULL,
    source_concept TEXT NOT NULL,
    target_concept TEXT NOT NULL,
    transfer_score REAL NOT NULL DEFAULT 0.5 CHECK (transfer_score >= 0.0 AND transfer_score <= 1.0),
    transfer_count INTEGER DEFAULT 0,
    last_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_doc, target_doc, source_concept, target_concept)
);

CREATE INDEX IF NOT EXISTS idx_concept_transfer_source_doc ON concept_transfer(source_doc);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_target_doc ON concept_transfer(target_doc);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_source_concept ON concept_transfer(source_concept);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_target_concept ON concept_transfer(target_concept);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_score ON concept_transfer(transfer_score);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_docs ON concept_transfer(source_doc, target_doc);
CREATE INDEX IF NOT EXISTS idx_concept_transfer_concepts ON concept_transfer(source_concept, target_concept);

-- Table: users (if not exists)
-- Base users table for foreign key references
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
