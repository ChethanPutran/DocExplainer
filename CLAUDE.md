# CLAUDE.md — Doc Explainer

A personal document tutor: upload a document (pdf, html, docx, txt, md, pptx), read it in a desktop app, and ask
questions in a chat. Answers are pitched to what the reader already knows. Three installable packages in one repo:

```
ui/   (doc_explainer_ui)   PySide6 desktop app   ──►  sdk/ (doc_explainer_sdk)  ──REST /v1 or JSON-RPC /rpc──►  src/doc_explainer (backend service)
```

The UI talks to the backend ONLY through the SDK; the backend runs on its own as a service. Details: `docs/ARCHITECTURE.md`.

---

## Commands

Use the ONE venv at the repo root (`.venv`); all three packages are installed in it in editable mode, so code
changes need a restart, not a reinstall (reinstall only if `pyproject.toml` dependencies/entry points change).
`ui/.venv` is stale and has no SDK: don't use it.

```bash
.venv/bin/doc-explainer-server                    # the backend (REST /v1, JSON-RPC /rpc), http://127.0.0.1:8000
.venv/bin/doc-explainer                           # the desktop UI
.venv/bin/doc-explainer-server --init-config      # create the (commented) override files in ~/.doc_explainer/config

.venv/bin/python -m pytest tests -q               # backend, ~3.5 min (339 tests). Real System + scripted LLM, no network
(cd sdk && ../.venv/bin/python -m pytest tests -q)                                   # SDK unit tests
(cd ui && QT_QPA_PLATFORM=offscreen PYTHONPATH=src:../sdk/src ../.venv/bin/python -m pytest tests -q)   # UI, ~20 s
```

Use `.venv/bin/python`; there is no global `python`. Keep stdout greps narrow: backend logs are noisy.

---

## Architecture (backend, `src/doc_explainer/`)

```
core/common      shared types + ports (Protocols)  ← the only thing core packages may share
core/document    parse by extension → ONE DocumentStructure (document→sections→paragraphs→sentences, tables, figures)
core/indexing    chunk + embed → Chroma vector index; structure store (structure only); content store (the text)
core/knowledge   temporal knowledge graph, built section by section, only earlier→later prerequisites, replayable
core/learner     learner model: BKT per concept, evidence, diagnostic planner
core/tutor       adaptation policy, prompts, streamed answers, quizzes, in-document + web resources
core/chat        sessions, messages, notes, summaries (per user AND document)
core/llm         provider wrappers + LLMFactory (shared clients, shared rate limiter)
orchestrator     composition root: adapters, IngestionPipeline, System, LLMJsonClient
api              FastAPI service: jobs, REST, JSON-RPC (one service layer), SSE streaming, http log
config           layered config (see below)
```

**Rule, enforced by `tests/test_core_independence.py`:** a core package imports only itself and `core.common`.
Each declares what it needs as a Protocol; `orchestrator/adapters.py` supplies implementations.

Key ideas
- **Time axis = reading-order `ordinal`.** Retrieval is filtered `ordinal <= position`; the knowledge graph is stamped with
  where each concept was introduced; nothing later than the reader's position can leak into an answer.
- **Knowledge graph** (`core/knowledge`): one LLM call per section; resumable (finished steps are skipped, failed ones retried
  in order); fails loudly (3 in a row or >30 %); document status `ready | partial | failed | none`.
- **Tutor** streams plain text + a `===FOLLOW-UPS===` block (`core/tutor/splitter.py` hides the marker). `explain/summarize/ask`
  just read the stream to the end. Events: status(understanding|searching|thinking|writing|finishing), delta, done.
- **LLM client** (`orchestrator/llm_client.py`): empty/cut-off answers are retried with a BIGGER `max_tokens` (×growth up to a
  cap), never the same request. Reasoning models (ling-3.0-flash, kimi) spend tokens thinking, so 1000 was too small.
  NOTE: `BaseLLM.generate` re-raises parser errors as `RuntimeError`; use `_is_format_error` (walks `__cause__`).
- **Jobs** (`api/service.py`, `api/jobs.py`): registration is a background job on ONE worker. States
  `queued → running → succeeded | failed | cancelled`, plus `paused`. Pause/cancel stop at the next step via the progress
  callback raising `JobStopped`; a paused job keeps its files and survives a restart; resume re-runs ingest on the same doc id
  (finished knowledge steps are skipped). Re-uploading a document whose graph is `failed`/`partial` resumes the SAME document.
  Cleanup never deletes a document that is already in the registry.
- **Privacy:** per-user via `X-User-Id`; optional bearer key; another user's ids look like "not found". Questions asked are
  stored only in chat sessions the reader creates; learner records keep concepts + question type, not the text.

### Frontend (`ui/src/doc_explainer_ui/`)
- `gui/windows/main_window.py` (tabs of documents, dock with the sidebar), `gui/managers/window_manager.py` (ALL service calls,
  on worker threads via `gui/utils/background.py`; results come back on the GUI thread).
- Sidebar tabs: **Explanation** (chat: streaming, sessions, notes, mic, speaker), **Resources**, **Learn & Quiz**.
- AI features appear only when the document is ready on the server (`MainWindow._refresh_ai`); until then the document is fully
  readable and the sidebar shows progress. Right-click menu on the viewers: Copy/Select all always; Explain/Summarize/Ask greyed
  until ready. PDF pages are images: the menu acts on the clicked page's text.
- Voice: `VoiceInput` is a hidden recogniser (Whisper, resident, idle-unload); the chat panel's mic button drives it and the words go
  into the input box. `SentenceSpeaker` (pyttsx3) reads sentences and announces each start, so text is revealed in step with the voice.
- Qt gotcha: `deleteLater`'d widgets must not be reused (create placeholders fresh). Workers touch only the hub QObject.

---

## Configuration: ONE source for every default

Defaults are written once, in packaged YAML; code never repeats them (`tests/test_config_layers.py` and
`ui/tests/test_config_layers.py` check that). Order: **packaged defaults → your file (only the keys you change) → environment.**

| Config | Defaults (packaged) | Your overrides (`~/.doc_explainer/config/`) | Env |
|---|---|---|---|
| LLM | `src/doc_explainer/config/defaults/llm_config.yaml` | `llm_config.yaml` | `DOC_EXPLAINER_LLM_<KEY>` |
| Backend | `.../defaults/backend_config.yaml` | `backend_config.yaml` | `DOC_EXPLAINER_<SECTION>_<KEY>` |
| UI | `ui/src/doc_explainer_ui/defaults/ui_config.yaml` | `ui_config.yaml` | `DOC_EXPLAINER_UI_<SECTION>_<KEY>` |
| Server process | `ApiSettings` (`api/settings.py`) | — | `DOC_EXPLAINER_HOST/PORT/API_KEY/DATA_DIR/...` |

- Default provider **airouter**, model **inclusionai/ling-3.0-flash**, 10 requests/min. Provider facts (model, endpoint, key
  variable) live in the `providers:` table of `llm_config.yaml`; `llm.model: null` = that provider's default.
- Secrets only from the environment / `.env` (`AIROUTER_API_KEY`, `OPENROUTER_API_KEY`, ...). `.env` is loaded in `config/layered.py`.
- User files are commented stubs, never full copies (so new defaults reach everyone). `save()` writes only differences.
- Embedding model loads from the local cache without contacting HuggingFace (`embeddings.check_for_updates: false`); chatty
  libraries are hidden below WARNING (`logging.quiet`).
- `OpenRouterWrapper` ignores the configured endpoint and hard-codes `reasoning={"effort": "high"}` (known, left alone).

## Added 2026-11-27 (all tested offscreen/with the scripted LLM, NOT yet clicked through in the live app)
- UI: Library window (list/open/delete, build or remove a document's knowledge graph), Recent menu (`gui/utils/recent_files.py`, `recent.json` also maps doc id -> local file),
  Settings > Learning (`knowledge.build_graph: always|ask|never`), Ctrl+F find bar, drag and drop, Ctrl+wheel zoom, Progress dashboard, Export chat (.md/.pdf),
  "Save to notes / Bookmark / Highlight" in the viewer menu, Piper neural voice (`voice.output_engine`, falls back to pyttsx3), themed QSS in `gui/styles/`.
- Backend/API/SDK: `DELETE /documents/{id}/knowledge`, `GET .../review` (spaced repetition), `GET .../history`, `GET .../sessions/{sid}/export`, re-registering a
  graph-less document builds the graph on the SAME document, deleting a document removes its stored upload, OCR for scanned PDFs (`core/document/ocr.py`, RapidOCR),
  tutor prompt asks for spoken-style prose with page citations, snake_case concept names become words.
- Known: `ui/tests/test_service_actions.py::test_a_question_streams_into_the_chat_for_the_located_section` is a timing flake when the machine is busy.
- Mic input needs sane system gain: with Mic Boost/Capture/Digital maxed the noise floor exceeds the speech trigger (the UI now warns).

## Where things are stored
- Server data: `~/.doc_explainer/server/` — `uploads/`, `documents.json`, `jobs.json`, and `system/` with `structure.db`,
  `content/<doc>.json`, `vectors/` (Chroma), `knowledge.db`, `learner.db`, `quiz.db`, `chat.db`. (`server/memory` is a leftover.)
- Logs: `~/.doc_explainer/logs/` — `app.log`, and `http_logs.log` (server requests: method, path, status, ms, bytes; no bodies, headers
  or queries). The UI writes its own `http_logs.log` (requests it sends) in the same folder.
- Config: `~/.doc_explainer/config/`. UI extras: `recent.json`, `cache/`.

## API surface (REST `/v1`, JSON-RPC `/rpc`; the SDK's `OPERATIONS` table drives both)
Documents, jobs (`pause|resume|cancel`), structure/knowledge/growth/concepts, learner/diagnostic/quiz/feedback/progress,
chat: `POST /documents/{id}/chat/stream` (SSE: session, status, delta, done|error), `/chat` (one piece), sessions + notes +
`/sessions/{sid}/summary`. JSON-RPC cannot stream: the SDK replays a non-streamed turn as events.
Add an operation in `sdk/src/doc_explainer_sdk/protocols.py` + `client.py` (sync AND async: a parity test enforces it),
`api/schemas.py`, `api/rest.py`, `api/rpc.py`, `api/service.py`, and the error table in `api/errors.py` (+ SDK `_BY_CODE`).

## Testing conventions
- Backend tests use the REAL `System` (stores, vector index, orchestration) with a scripted LLM (`tests/fakes.py`: `Brain`,
  `HashingEmbedder`). `Brain.stream_text` returns a canned reply with the follow-up block; `gate`/`stream_gate` events hold a
  step open for pause/cancel tests.
- Tests of changed behaviour should fail without the fix (several were checked that way). Don't weaken a test to pass it.
- UI tests run offscreen; WindowManager is built with `object.__new__` and fake pieces; use the `pump` fixture to process Qt events.

## State and caveats (as of 2026-11-27)
- Everything above is implemented and its suites pass; NOTHING is committed (changes are staged with `git add -A`).
- Verified live: the model streams through `LLMJsonClient.stream_text` (ling-3.0-flash); the server logs requests to `http_logs.log`;
  the embedder loads with no HuggingFace requests. NOT verified in the running app with real hardware: microphone + Whisper,
  pyttsx3 speech, the full chat/voice flow in the desktop window.
- Concept-extraction quality on a real book is untested end to end with the final settings (earlier failures were the 1000-token limit;
  re-upload resumes only the failed sections).
- DOCX/PPTX open in `OfficeViewer` (converted to HTML by `gui/utils/office_html.py`); text-like viewers find the reader's section by heading position (`section_locator.py`), PDFs by page.
- Stray junk in the repo root: a file named `=7.4.0` (from an unquoted `pip install pkg>=7.4.0`); safe to delete.
- Old `ui/src/doc_explainer_ui/widgets/` (not `gui/widgets`) looks like dead code from the monolith.
