# 📘 Project: Doc Explainer

## A Context-Aware, Knowledge-Adaptive Document Tutor Using Large Language Models

---

## 1. Overview

**Doc Explainer** is an AI-augmented, document-centric intelligent tutoring system designed to deliver personalized, dependency-aware explanations of technical and educational content.

The system:

1. Explains **user-selected text** in context
2. Maintains **full-document understanding**
3. Builds and updates a **user knowledge model**
4. Adapts explanation **depth dynamically**
5. Supports **voice interaction**
6. Recommends **prerequisites or deeper material**

The project focuses on modular development of an adaptive tutoring architecture that combines scalable document retrieval, structured concept modeling, and multimodal interaction. The system is designed for both rigorous experimentation and real-world deployment.

---

## 2. Problem Definition

Given:

### 1. Document Structure

A hierarchical document tree:

```
Chapter
 └── Section
      └── Paragraph
           └── Sentence
```

### 2. User Interaction

* The user stops at a paragraph.
* The system:

  * Identifies the selected sentence.
  * Retrieves the corresponding Paragraph ID and Section ID.
  * Extracts the selected text span.

### 3. Concept Dependencies

* The paragraph may depend on concepts introduced earlier.
* These dependencies are tracked through a concept graph.

### 4. User Knowledge State

* The system maintains:

  * Which chapters/concepts the user knows.
  * Optionally, how well they know them.

### 5. Goal

Generate an explanation of the selected paragraph that:

* Uses only concepts the user already knows.
* Re-explains missing prerequisites if needed (user-controlled).
* Matches the appropriate depth for the user.

This results in:

> Personalized, dependency-aware, concept-grounded explanation generation.

---

## 3. Key Functionalities

### 3.1 Selected Text Explanation (Core Feature)

**Inputs:**

* Selected text span
* Surrounding section context
* Entire document summary
* User knowledge state

The explanation is contextual and document-grounded, not generic paraphrasing.

---

### 3.2 Full-Document Context Awareness

The system uses hierarchical memory:

```
Document
├── Section summaries
│     ├── Paragraph embeddings
│           ├── Sentence embeddings
│
├── Concept map
│     ├── Concept → Sections
│
├── Knowledge map
```

#### Methods

* Recursive chunking
* Section-level embeddings
* Concept extraction using LLM + NER

This enables:

* Cross-referencing within the document
* Reduced hallucination
* Structured retrieval

---

## 4. Differentiation from Existing Systems

| System       | Limitation                        |
| ------------ | --------------------------------- |
| ChatGPT      | No persistent user knowledge      |
| Kindle hints | Static, non-personal explanations |
| Khan Academy | Linear curriculum structure       |
| Copilot      | Task-oriented, not pedagogical    |

Doc Explainer is:

> Dynamic + Personalized + Dependency-Aware

---

## 5. Implemented Tasks

* Concept graph extraction
* Adaptive explanation depth
* Prerequisite suggestion (recommendation)
* LLM-based explanation generation
* Automatic concept extraction
* Embedding-based concept linking
* Persistent user knowledge store

---

## 6. Pending Tasks

* Context-aware RAG
* Knowledge mastery estimation
* Simple user profile (known / unknown concepts)
* Personalized curriculum generation
* Quiz-based feedback loop
* Reinforcement learning for tutoring policy
* Multi-document knowledge transfer
* Manual concept graph support
* Paragraph → concept tagging
* Explanation quality feedback loop
* RL-based explanation depth tuning
* Multimodal support (equations + diagrams)

---

## 7. System Modules

### 7.1 Core Module

#### Document

* Document caching
* Document processing

#### Memory

* Long-term memory
* Knowledge caching

#### Knowledge Modeling

* Concept extraction
* Knowledge graph / Concept graph
* User knowledge model

#### Explanation Engine

* Adaptive explainer
* Resource recommender
* Voice interface

#### Evaluation

* Knowledge evaluator

#### UI

* GUI
* Web application

---
