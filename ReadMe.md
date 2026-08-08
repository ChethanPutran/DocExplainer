# 📘 Doc Explainer

## A Context-Aware, Knowledge-Adaptive Document Tutor Using Large Language Models

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/yourusername/doc-explainer)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/doc-explainer/pulls)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Doc Explainer** is an AI-augmented, document-centric intelligent tutoring system designed to deliver personalized, dependency-aware explanations of technical and educational content. Unlike generic AI assistants, Doc Explainer:

- 📚 **Understands your documents** hierarchically (chapters → sections → paragraphs → sentences)
- 🧠 **Learns your knowledge state** and adapts explanations accordingly
- 🔗 **Maps concept dependencies** to ensure you understand prerequisites
- 🎯 **Generates context-aware explanations** grounded in the actual document
- 🗣️ **Supports voice interaction** for hands-free learning

The system combines scalable document retrieval, structured concept modeling, and multimodal interaction to create a truly personalized tutoring experience.

---

## ✨ Key Features

### 📖 Document Understanding
| Feature | Description |
|---------|-------------|
| **Hierarchical Parsing** | Processes documents into chapters, sections, paragraphs, and sentences |
| **Multi-format Support** | PDF, TXT, HTML, Markdown, DOCX, PPTX |
| **Concept Extraction** | Automatically identifies and extracts key concepts using LLM + NER |
| **Knowledge Graph** | Builds a dependency graph of concepts within the document |
| **Semantic Search** | Vector-based retrieval of relevant sections |

### 🧠 Personalized Learning
| Feature | Description |
|---------|-------------|
| **User Knowledge Model** | Tracks which concepts the user knows and confidence levels |
| **Bayesian Knowledge Tracing** | Updates knowledge state based on interactions |
| **Adaptive Explanations** | Adjusts depth and complexity based on user expertise |
| **Prerequisite Detection** | Identifies and recommends prerequisite concepts |
| **Learning Path Generation** | Creates personalized curriculum based on knowledge gaps |

### 🎙️ Interaction Modes
| Mode | Description |
|------|-------------|
| **Text Selection** | Select any text to get instant explanation |
| **Question Answering** | Ask questions about the document |
| **Voice Input** | Speak your questions naturally |
| **Voice Output** | Listen to explanations hands-free |
| **Follow-up Questions** | Suggested questions based on context |

### 📊 Visualization
- Interactive knowledge graphs
- Document structure visualization
- User knowledge heatmaps
- Learning progress tracking

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Main UI   │  │    Voice    │  │   Document Viewer   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Pipeline  │  │   Context   │  │      Services       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Core Modules                            │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│  Document   │   Memory    │  Knowledge  │   Explanation    │
│  Processor  │   Manager   │   Graph     │     Engine       │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│  Parser     │ Long-term   │ Concept     │ Adaptive         │
│  Builder    │ Session     │ Extraction  │ Explainer        │
│  Cacher     │ Context     │ Relations   │ Recommender      │
└─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Document   │  │   User      │  │   Knowledge Graph   │ │
│  │  Store      │  │   Store     │  │      Store          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Module Details

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| **Document** | Document processing and management | `DocumentParser`, `DocumentTree`, `DocumentEngine` |
| **Memory** | User and session memory management | `LongTermMemory`, `SessionManager`, `Context` |
| **Knowledge** | Concept extraction and graph management | `ConceptExtractor`, `KnowledgeGraph`, `PrerequisiteAnalyzer` |
| **Explanation** | Adaptive explanation generation | `AdaptiveExplainer`, `ResourceRecommender` |
| **Agent** | LLM interaction and prompt management | `LLMWrapper`, `PromptTemplates`, `OutputParsers` |
| **Orchestrator** | Pipeline coordination | `DocExplainerOrchestrator`, `Pipelines` |
| **UI** | Graphical user interface | `MainWindow`, `Sidebar`, `Viewers` |

---

## 🔄 How It Works

### 1. Document Processing Pipeline

```
Raw Document → Parser → Document Tree → Chunking → Embeddings → Vector Store
                        ↓
                  Knowledge Graph ← Concept Extraction
```

### 2. User Interaction Flow

```
User Selects Text → Context Resolution → Section ID → Document Context
                                          ↓
                                 User Knowledge State
                                          ↓
                            Adaptive Explanation Generation
                                          ↓
                         Follow-up Questions + Resource Recommendations
```

### 3. Knowledge Tracing

```
Initial State → Interaction → Bayesian Update → New Knowledge State
                  ↑                ↓
              User Feedback   Confidence Score
```

### 4. Explanation Generation

```
Selected Text + Document Context + User Knowledge → Prompt Engineering
                                                   ↓
                                            LLM Generation
                                                   ↓
                                     Explanation + Follow-up Questions
                                                   ↓
                                      Resource Enrichment (Videos/Articles)
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Git** (for cloning repository)
- **System dependencies** (see platform-specific instructions)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/doc-explainer.git
cd doc-explainer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Platform-Specific Installation

<details>
<summary><b>🐧 Linux (Ubuntu/Debian)</b></summary>

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
    libportaudio2 \
    libsqlite3-dev \
    libbz2-dev \
    libreadline-dev

# Install spaCy model
python -m spacy download en_core_web_sm

# Install with all extras
pip install -e .[full]
```
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install \
    poppler \
    tesseract \
    portaudio \
    libxml2 \
    libxslt \
    python@3.10

# Install spaCy model
python -m spacy download en_core_web_sm

# Install with MLX support for Apple Silicon
pip install -e .[mlx,full]
```
</details>

<details>
<summary><b>🪟 Windows</b></summary>

```powershell
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install spaCy model
python -m spacy download en_core_web_sm

# Install with all extras
pip install -e .[full]
```

**Note:** For PyAudio on Windows, you may need to install a pre-compiled wheel:
```powershell
pip install pipwin
pipwin install pyaudio
```
</details>

### Docker Installation

```bash
# Build Docker image
docker build -t doc-explainer .

# Run container
docker run -v /path/to/documents:/documents doc-explainer /documents/paper.pdf
```

### Development Installation

```bash
# Install with development dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

---

## 🚀 Usage Guide

### Command Line Interface

```bash
# Basic usage
doc-explainer
doc-explainer /path/to/document.pdf

# With theme
doc-explainer --theme dark document.pdf

# With custom config
doc-explainer --config ~/.doc_explainer/config.json document.pdf

# Debug mode
doc-explainer --debug document.pdf

# Disable splash screen
doc-explainer --no-splash document.pdf

# Set log level
doc-explainer --log-level DEBUG document.pdf

# Profile performance
doc-explainer --profile document.pdf
```

### GUI Application

#### Main Window

![Main Window](docs/images/main_window.png)

1. **Document Viewer** (center) - Displays the current document
2. **AI Tutor Sidebar** (right) - Shows explanations and recommendations
3. **Toolbar** (top) - Document controls and settings
4. **Voice Controls** - Voice input/output buttons
5. **Tab Bar** - Multiple documents support

#### Basic Workflow

1. **Open Document**
   - Click "Open Document" button or press `Ctrl+O`
   - Select PDF, TXT, HTML, or other supported format

2. **Select Text**
   - Highlight any text in the document
   - Right-click or use toolbar buttons

3. **Choose Action**
   - **Explain** (`Ctrl+E`) - Get detailed explanation
   - **Summarize** (`Ctrl+S`) - Get concise summary
   - **Ask** (`Ctrl+Q`) - Ask a question

4. **Interact with Results**
   - Read explanation in sidebar
   - Click follow-up questions for deeper understanding
   - Access recommended resources (videos, articles)
   - Use voice for hands-free interaction

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open document |
| `Ctrl+W` | Close current tab |
| `Ctrl+Tab` | Switch tabs |
| `Ctrl+E` | Explain selected text |
| `Ctrl+S` | Summarize selected text |
| `Ctrl+Q` | Ask question |
| `Ctrl+F` | Find in document |
| `Ctrl++` | Zoom in |
| `Ctrl+-` | Zoom out |
| `Ctrl+0` | Reset zoom |
| `Ctrl+B` | Toggle sidebar |
| `F5` | Refresh document |
| `F1` | Help |
| `Ctrl+,` | Settings |
| `Ctrl+Q` | Quit |

### Voice Commands

Enable voice input and try these commands:

- "Explain this paragraph"
- "Summarize this section"
- "What does [term] mean?"
- "Give me an example"
- "Simplify that"
- "Tell me more about [concept]"

### Example Sessions

#### Session 1: Learning a New Concept

```
User: Opens a machine learning paper
User: Selects "Transformer architecture"
System: Explains using beginner-level language, noting it requires understanding of attention
User: Clicks follow-up "What is attention?"
System: Explains attention mechanism, updates knowledge state
User: Asks "Show me an example"
System: Provides concrete example with visualization
```

#### Session 2: Reviewing Known Material

```
User: Selects same text a week later
System: Recognizes user now knows attention, provides advanced explanation
User: Asks "How does this compare to RNNs?"
System: Draws comparisons based on known concepts
```

---

## ⚙️ Configuration

### Configuration File

Create `~/.doc_explainer/config/config.json`:

```json
{
    "theme": "light",
    "llm_provider": "gemini",
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
    "openai_api_key": "YOUR_OPENAI_API_KEY",
    "window": {
        "width": 1200,
        "height": 800,
        "maximized": false
    },
    "sidebar": {
        "visible": true,
        "width": 350,
        "position": "right"
    },
    "voice": {
        "enabled": true,
        "input_device": "default",
        "output_enabled": true,
        "output_rate": 150,
        "output_volume": 0.9
    },
    "documents": {
        "default_zoom": 1.0,
        "max_recent": 10,
        "auto_save_interval": 5
    },
    "cache": {
        "enabled": true,
        "size_mb": 500,
        "location": "~/.doc_explainer/cache"
    },
    "llm": {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "temperature": 1.0,
        "max_tokens": null
    },
    "knowledge_graph": {
        "enabled": true,
        "auto_build": true
    },
    "memory": {
        "enabled": true,
        "session_tracking": true
    },
    "startup": {
        "open_last_docs": true,
        "check_updates": true,
        "show_splash": true
    },
    "debug": {
        "enabled": false,
        "log_level": "INFO",
        "profiling": false
    }
}
```

### Environment Variables

Create `.env` file in project root:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Configuration
DOC_EXPLAINER_THEME=light
DOC_EXPLAINER_DEBUG=false
DOC_EXPLAINER_LOG_LEVEL=INFO

# Paths
DOC_EXPLAINER_CACHE_DIR=~/.doc_explainer/cache
DOC_EXPLAINER_CONFIG_DIR=~/.doc_explainer/config
DOC_EXPLAINER_DATA_DIR=~/.doc_explainer/data

# LLM Settings
DOC_EXPLAINER_LLM_PROVIDER=gemini
DOC_EXPLAINER_LLM_TEMPERATURE=1.0
DOC_EXPLAINER_LLM_MAX_TOKENS=4096

# Feature Flags
DOC_EXPLAINER_ENABLE_VOICE=true
DOC_EXPLAINER_ENABLE_KG=true
DOC_EXPLAINER_ENABLE_MEMORY=true
```

### Command Line Arguments

```bash
doc-explainer --help
```

```
usage: doc-explainer [-h] [--config CONFIG] [--theme {light,dark,high_contrast,sepia}]
                     [--debug] [--no-splash] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                     [--version] [--profile] [--reset-config] [--clear-cache]
                     [document]

Doc Explainer - Intelligent Document Explanation System

positional arguments:
  document              Document to open (PDF, TXT, HTML, etc.)

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Configuration file path
  --theme {light,dark,high_contrast,sepia}, -t {light,dark,high_contrast,sepia}
                        Theme to use
  --debug, -d           Enable debug mode
  --no-splash           Disable splash screen
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set logging level
  --version, -v         show program's version number and exit
  --profile             Enable performance profiling
  --reset-config        Reset configuration to defaults
  --clear-cache         Clear application cache
```

---

## 📚 API Reference

### Core Classes

#### `DocumentProcessor`

```python
from src.core.document import DocumentProcessor

processor = DocumentProcessor()
doc_id = processor.load_document("path/to/file.pdf")
tree = processor.build_document_tree(doc_id)
```

#### `AdaptiveExplainer`

```python
from src.core.explanation_engine import AdaptiveExplainer

explainer = AdaptiveExplainer(style="intermediate")
explanation = explainer.explain(text, context)
summary = explainer.summarize(text, context)
answer = explainer.ask(question, context)
```

#### `KnowledgeGraph`

```python
from src.core.knowledge import KnowledgeGraph

kg = KnowledgeGraph()
kg.add_concept("attention")
kg.add_relationship("transformer", "depends_on", "attention")
prereqs = kg.get_prerequisites("transformer")
```

#### `UserManager`

```python
from src.core.user import UserManager

user_manager = UserManager("user123")
state = user_manager.get_user_knowledge()
user_manager.update_knowledge({"concept": "attention", "correct": True})
```

### REST API (if enabled)

```bash
# Register document
POST /api/documents
{"path": "/path/to/file.pdf"}

# Get explanation
POST /api/explain
{
    "doc_id": "123",
    "text": "selected text",
    "section_id": 5
}

# Ask question
POST /api/ask
{
    "doc_id": "123",
    "question": "What is attention?",
    "section_id": 5
}

# Get user state
GET /api/user/{user_id}/knowledge
```

---

## 🛠️ Development

### Project Structure

```
doc-explainer/
├── src/
│   ├── core/               # Core modules
│   │   ├── document/       # Document processing
│   │   ├── memory/         # Memory management
│   │   ├── knowledge/      # Knowledge graph
│   │   ├── explanation/    # Explanation engine
│   │   ├── user/           # User modeling
│   │   └── agent/          # LLM interaction
│   ├── orchestrator/       # Pipeline coordination
│   ├── store/              # Data persistence
│   ├── ui/                 # User interface
│   │   ├── gui/            # Qt GUI
│   │   └── web/            # Web interface (optional)
│   └── cli/                # Command line interface
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── examples/                # Example notebooks
├── scripts/                 # Utility scripts
├── data/                    # Data directory
├── config/                  # Configuration files
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── README.md                # This file
└── LICENSE                  # MIT License
```

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/doc-explainer.git
cd doc-explainer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Build documentation
cd docs
make html
```

### Code Style

This project follows:
- **PEP 8** for Python code
- **Black** for formatting
- **isort** for import sorting
- **mypy** for type checking
- **pylint** for linting

```bash
# Format code
black src/
isort src/

# Type check
mypy src/

# Lint
pylint src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_document.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run GUI tests (headless)
pytest tests/test_gui.py --qt
```

### Building Distributions

```bash
# Build source and wheel distributions
python -m build

# Upload to PyPI (if publishing)
twine upload dist/*
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Areas

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 UI/UX enhancements
- 🌐 Internationalization
- ⚡ Performance optimizations
- 🧪 Test coverage

### Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

TO BE FILLED: Add license summary and attribution.

---

## 🙏 Acknowledgments

- **LangChain** community for LLM orchestration
- **PySide6** team for Qt bindings
- **Google Gemini** and **OpenAI** for LLM APIs
- **spaCy** for NLP capabilities
- **NetworkX** for graph algorithms
- All open-source contributors

---

## 📞 Contact & Support

- 📧 **Email**: support@docexplainer.com
- 📚 **Documentation**: [https://docs.docexplainer.com](https://docs.docexplainer.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/doc-explainer/issues)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/doc-explainer&type=Date)](https://star-history.com/#yourusername/doc-explainer&Date)

---

**Built with ❤️ for lifelong learners everywhere**
