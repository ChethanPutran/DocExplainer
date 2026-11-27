# 📘 Doc Explainer

**A personal document tutor.** Open a document, read it in a desktop app, and ask questions in a chat. Answers are pitched
to what *you* already know: the app builds a map of the concepts in the document, tracks what you have learned, and
explains accordingly. It can also read answers aloud, quiz you, and remind you what you are likely to forget.

```
Desktop UI (PySide6)  ──►  SDK  ──REST /v1  or  JSON-RPC /rpc──►  Backend service (FastAPI)
  ui/  doc_explainer_ui      sdk/  doc_explainer_sdk              src/doc_explainer
```

The UI talks to the backend **only through the SDK**, and the backend runs on its own as a service, so you can also use
the SDK from scripts or put another front end on the same API.

---

## Contents
- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Configuration](#configuration)
- [The SDK and the API](#the-sdk-and-the-api)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)
- [Known limits](#known-limits)

---

## What it does

**Reading**
- Open **PDF, Word (.docx), PowerPoint (.pptx), HTML, Markdown and text** files. The document is readable immediately,
  while the server processes it in the background.
- **Zoom** with the toolbar, `Ctrl +` / `Ctrl -` / `Ctrl 0`, or **Ctrl + mouse wheel**.
- **Find** in the document with `Ctrl+F` (`F3` / `Shift+F3` for next / previous); PDF matches are highlighted on the page.
- **Drag and drop** files onto the window. A **Recent** menu lists what you opened lately.
- Right-click: copy, **Explain**, **Summarize**, **Ask**, **Save to notes**, **Bookmark this place**, **Highlight**.

**Learning**
- **Chat tutor**: streamed answers in a conversation with sessions, notes and summaries. Answers use plain, spoken-style
  language and name the section and page they come from.
- **Adapts to you**: a knowledge graph of the document's concepts plus a model of what you know decides what to explain
  and how deeply. Nothing later than your reading position can leak into an answer.
- **Quizzes**: a diagnostic to place your level, and practice questions per section.
- **Progress dashboard**: your level, mastery per concept, a progress-over-time chart, and a **review list**
  (spaced repetition: concepts you are likely forgetting).
- **Resources**: related passages from the document and from the web.

**Library and control**
- **Library**: every processed document, with search; open it, delete it, build or remove its knowledge graph.
- **Knowledge graph is optional**: Settings → Learning → *Always / Ask for each file / Never*. Without it you can still read
  and ask questions; with it you get adaptive explanations, quizzes and knowledge tracking. You can build or remove it
  per document later from the Library.
- **Export** a conversation and its notes as Markdown or PDF.

**Voice**
- **Speak your questions** (Whisper, runs locally) and **hear the answers** read aloud. Output uses the free offline
  **Piper** neural voice when installed, falling back to the system voice. Markdown symbols are never read out.

**Other**
- Scanned PDFs are read with **OCR** (RapidOCR, offline).
- Light, dark and high-contrast themes.
- Private per user (`X-User-Id`); your questions are stored only in chat sessions you create.

---

## Quick start

**Requirements:** Python 3.10+ (3.12 recommended), Linux/macOS/Windows, an API key for a language model provider.

```bash
git clone https://github.com/ChethanPutran/DocExplainer.git
cd DocExplainer

# Linux only: system libraries for the microphone
sudo apt install -y portaudio19-dev python3-dev

# One virtual environment for all three packages
python -m venv .venv
.venv/bin/python -m pip install -e ./sdk -e . -e ./ui
```

Give the backend a model key (the default provider is **airouter**; others are listed in `llm_config.yaml`):

```bash
echo 'AIROUTER_API_KEY=your-key' >> .env          # or export it in your shell
```

Run the two parts (separate terminals):

```bash
.venv/bin/doc-explainer-server        # backend: http://127.0.0.1:8000  (REST /v1, JSON-RPC /rpc, docs at /docs)
.venv/bin/doc-explainer               # desktop app
.venv/bin/doc-explainer path/to/file.pdf      # …or open a file directly
```

**Optional: a natural voice (Piper)**

```bash
.venv/bin/python -m pip install piper-tts
mkdir -p ~/.doc_explainer/voices
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir ~/.doc_explainer/voices
```

Without it the app uses the system voice (`pyttsx3`). Other voices: `en_US-amy-medium`, `en_GB-alba-medium`, …
(set `voice.piper_voice` in the UI config).

**Docker:** the `Dockerfile` uses `uv`; run `uv lock` first so `uv.lock` matches `pyproject.toml`.

---

## Using the app

| Toolbar | What it does |
|---|---|
| 📂 **Open** / 🕘 **Recent** | Open a file, or reopen a recent one |
| 📚 **Library** | All processed documents: open, delete, build / remove the knowledge graph |
| 🔍 **Zoom** | Zoom in / out (also `Ctrl` + wheel) |
| 🧠 **Quiz me** | Diagnostic and practice quizzes |
| 📈 **Progress** | Mastery, history and what to review |
| ⬇ **Export chat** | Save the conversation as `.md` or `.pdf` |
| 📋 **Job status** | Processing progress; pause, resume or cancel |
| 🌓 **Theme** | Switch theme |

The sidebar has three tabs: **Explanation** (chat, sessions, notes, microphone 🎤 and speaker 🔊), **Resources** and
**Learn & Quiz**. The tutor becomes available once the server has finished processing the document (a progress marker
shows on the tab); until then the document is fully readable.

**Voice tips:** press 🎤 and speak; the words land in the input box. Press 🔊 to have answers read aloud, with the text
appearing as it is spoken. If the microphone seems deaf, lower the input gain in your system sound settings: with the
boost at maximum the background noise can exceed your speech and nothing is picked up (the app warns about this).

---

## Configuration

One source for every default: packaged YAML. **Order: packaged defaults → your file (only the keys you change) → environment.**

| Config | Your overrides (`~/.doc_explainer/config/`) | Environment |
|---|---|---|
| Language model | `llm_config.yaml` | `DOC_EXPLAINER_LLM_<KEY>` |
| Backend | `backend_config.yaml` | `DOC_EXPLAINER_<SECTION>_<KEY>` |
| UI | `ui_config.yaml` | `DOC_EXPLAINER_UI_<SECTION>_<KEY>` |
| Server process | flags | `DOC_EXPLAINER_HOST / PORT / API_KEY / DATA_DIR …` |

Create the commented override files with `doc-explainer-server --init-config` (and `doc-explainer --init-config` for the UI).
Secrets come only from the environment or `.env` (`AIROUTER_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`…).

Useful settings:

```yaml
# ui_config.yaml
knowledge:
  build_graph: always        # always | ask | never
voice:
  output_engine: auto        # auto (Piper if installed) | piper | system
  piper_voice: en_US-lessac-medium
  output_rate: 150
documents:
  default_zoom: 1.0
  max_recent_files: 10
```

```yaml
# llm_config.yaml
llm:
  provider: airouter         # or openrouter, openai, gemini, ollama
  requests_per_minute: 10
```

```yaml
# backend_config.yaml
knowledge_graph:
  enabled: true              # server-wide switch for building graphs
```

**Where things are stored:** `~/.doc_explainer/` — `server/` (uploads, documents, jobs and the `system/` databases: structure,
vectors, knowledge graph, learner, quizzes, chats), `logs/` (`app.log`, `http_logs.log`), `config/`, `voices/`, `recent.json`.

---

## The SDK and the API

```python
from doc_explainer_sdk import DocExplainerClient     # AsyncDocExplainerClient is the awaitable twin

client = DocExplainerClient("http://127.0.0.1:8000")          # protocol="rest" (default) or "rpc"
doc = client.register_document("paper.pdf")                   # waits for processing; build_graph=False to skip the graph
print(client.ask(doc.doc_id, "What is the main finding?").text)

for event in client.chat_stream(doc.doc_id, "Explain the Calvin cycle"):    # streamed (REST)
    if event.type == "delta":
        print(event.text, end="")

client.get_review(doc.doc_id)          # concepts you are likely forgetting
client.export_session(doc.doc_id, session_id).markdown
client.delete_knowledge(doc.doc_id)    # drop the graph and your tracking; the document stays
```

REST lives under `/v1` and JSON-RPC at `/rpc` (one service layer behind both; interactive docs at `/docs`). Main areas:

| Area | Operations |
|---|---|
| Documents | upload, list, get, delete, structure, section lookup, context |
| Jobs | list, get, `pause` / `resume` / `cancel` (registration is a background job) |
| Knowledge | knowledge graph, growth, concepts, prerequisites, learning path, **delete graph** |
| Learner | learner state, diagnostic, quizzes, feedback, progress, **review**, **history** |
| Chat | streamed chat (SSE), sessions, notes, summaries, **export** |

Users are separated by the `X-User-Id` header; an API key (`DOC_EXPLAINER_API_KEY`) is required to listen on a
non-loopback address. The server speaks plain HTTP: put TLS in front of it.

---

## How it works

```
upload ─► parse (pdf/docx/pptx/html/md/txt, OCR for scans) ─► one DocumentStructure
          (document → sections → paragraphs → sentences, tables, figures)
        ─► chunk + embed ─► vector index          (text store + structure store)
        ─► knowledge graph, one LLM call per section (optional)

question ─► where are you reading? ─► retrieve only text up to that point
         ─► what do you know?      ─► adapt depth, define new terms, skip the known
         ─► stream an answer, then 3 follow-up questions
```

- **Time axis = reading order.** Retrieval is filtered to the reader's position and the graph is stamped with where each
  concept was introduced, so nothing later than the reader's position can appear in an answer.
- **Knowledge graph:** built section by section, only earlier→later prerequisites, resumable (finished steps are kept,
  failed ones retried) and loud about failures. Status: `ready | partial | failed | none`.
- **Learner model:** a belief per concept (Bayesian knowledge tracing) updated by quiz answers, question types and
  feedback. Review scheduling decays that belief over time, more slowly the more you have practised a concept.
- **Jobs:** processing runs on one background worker (`queued → running → succeeded | failed | cancelled`, plus `paused`).
  A paused job survives a restart; re-uploading a document whose graph failed or is partial resumes the same document, and
  so does asking for a graph on a document that had none.

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`CLAUDE.md`](CLAUDE.md).

---

## Project layout

```
sdk/   doc_explainer_sdk        typed client (sync + async), REST and JSON-RPC protocols; depends only on httpx, pydantic
ui/    doc_explainer_ui         PySide6 desktop app; talks to the backend only through the SDK
src/doc_explainer/
  core/common      shared types + ports (Protocols): the only thing core packages may share
  core/document    parsers → one DocumentStructure
  core/indexing    chunking, embeddings, vector index, stores
  core/knowledge   temporal knowledge graph
  core/learner     learner model, evidence, diagnostics
  core/tutor       adaptation policy, prompts, streaming, quizzes, resources
  core/chat        sessions, notes, summaries
  core/llm         provider wrappers (airouter, openrouter, openai, gemini, ollama)
  orchestrator     composition root: adapters, ingestion pipeline
  api              FastAPI service: jobs, REST, JSON-RPC, streaming
  config           layered config
tests/ sdk/tests/ ui/tests/
```

A core package imports only itself and `core.common`; `tests/test_core_independence.py` enforces it.

---

## Development and tests

Use the one virtual environment at the repo root. All three packages are installed in editable mode, so code changes
need a restart, not a reinstall (reinstall only when dependencies or entry points change).

```bash
.venv/bin/python -m pytest tests -q                                                  # backend (~2.5 min), real System + scripted LLM, no network
(cd sdk && ../.venv/bin/python -m pytest tests -q)                                   # SDK
(cd ui && QT_QPA_PLATFORM=offscreen PYTHONPATH=src:../sdk/src ../.venv/bin/python -m pytest tests -q)   # UI, offscreen
```

Adding an API operation: `sdk/.../protocols.py` + `client.py` (sync **and** async — a parity test enforces it),
`api/schemas.py`, `api/rest.py`, `api/rpc.py`, `api/service.py`, and the error table in `api/errors.py`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot reach the Doc Explainer server" | Start `doc-explainer-server` and check the URL in Settings → Server |
| Documents stay "processing" or the graph fails | Check the key (`AIROUTER_API_KEY`) and `~/.doc_explainer/logs/app.log`; re-uploading resumes only the failed sections |
| Microphone captures nothing | Lower the input gain / "Mic Boost" in system sound settings; the app warns when the noise level is too high |
| Robotic voice | Install Piper (see Quick start) |
| Scanned PDF gives "no readable text" | OCR needs `rapidocr-onnxruntime` (installed with the backend); very poor scans may still fail |
| Docker build fails on `--frozen` | Run `uv lock` to refresh `uv.lock` |

---

## Known limits

- DOCX/PPTX are shown as formatted text (images are not rendered).
- A document's knowledge graph can be removed or rebuilt, but highlights are kept only while the document is open.
- Concept-extraction quality on very large books has not been tested end to end with the current settings.
- Reasoning models spend tokens thinking; if answers are cut off, raise `llm.max_tokens`.

---

## License

MIT — see [`LICENSE`](LICENSE).
