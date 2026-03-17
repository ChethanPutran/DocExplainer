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


## Installation Instructions

### 1. Basic Installation (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/doc-explainer.git
cd doc-explainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### 2. Installation from Source

```bash
# Build the package
python -m build

# Install the built package
pip install dist/doc-explainer-1.0.0.tar.gz
```

### 3. Installation with pip (from repository)

```bash
# Direct install from git
pip install git+https://github.com/yourusername/doc-explainer.git

# Install with specific extras
pip install git+https://github.com/yourusername/doc-explainer.git#egg=doc-explainer[full]
```

### 4. Running after Installation

```bash
# Run as installed package
doc-explainer
doc-explainer /path/to/document.pdf
doc-explainer --theme dark document.pdf

# Run as module
python -m src.ui.gui.app

# Run with Python directly
python run.py
```

### 5. Platform-Specific Setup

#### Linux
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    poppler-utils \
    tesseract-ocr \
    portaudio19-dev \
    libportaudio2

# Install spaCy model
python -m spacy download en_core_web_sm
```

#### macOS
```bash
# Install system dependencies
brew install \
    poppler \
    tesseract \
    portaudio \
    libxml2 \
    libxslt

# Install spaCy model
python -m spacy download en_core_web_sm
```

#### Windows
```powershell
# Install spaCy model
python -m spacy download en_core_web_sm

# Note: You may need to install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 6. Docker Installation

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    tesseract-ocr \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

RUN pip install -e .

CMD ["doc-explainer"]
```

Build and run:

```bash
docker build -t doc-explainer .
docker run -v /path/to/documents:/documents doc-explainer /documents/paper.pdf
```

### 7. Configuration File

Create `~/.doc_explainer/config/config.json`:

```json
{
    "theme": "light",
    "llm_provider": "gemini",
    "gemini_api_key": "YOUR_API_KEY_HERE",
    "openai_api_key": "YOUR_API_KEY_HERE",
    "window_width": 1200,
    "window_height": 800,
    "sidebar_visible": true,
    "voice_enabled": true
}
```

### 8. Environment Variables

Create `.env` file:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Configuration
DOC_EXPLAINER_THEME=light
DOC_EXPLAINER_DEBUG=false
DOC_EXPLAINER_LOG_LEVEL=INFO

# Paths
DOC_EXPLAINER_CACHE_DIR=~/.doc_explainer/cache
DOC_EXPLAINER_CONFIG_DIR=~/.doc_explainer/config
```

Now you have a complete, production-ready setup for your Doc Explainer application!