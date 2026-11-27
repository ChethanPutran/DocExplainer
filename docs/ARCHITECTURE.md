# Architecture

Three installable packages; the UI never touches the backend directly.

```
ui/ (doc_explainer_ui)  ──►  sdk/ (doc_explainer_sdk)  ──REST /v1 or JSON-RPC /rpc──►  backend (doc_explainer)
```

## Backend layers (`src/doc_explainer/`)

```
core/common        shared types + ports (Protocols)           ◄── the only thing core packages may share
core/document      parse by extension → one DocumentStructure
core/indexing      chunk + embed → vector index; structure store; content store
core/knowledge     temporal knowledge graph, built section by section
core/learner       learner model: BKT, evidence, diagnostic planner
core/tutor         adaptation policy, prompts, resources, quizzes
core/llm           provider wrappers + shared-client factory
orchestrator       composition root: adapters, IngestionPipeline, System
api                FastAPI service: jobs, REST, JSON-RPC (one service layer)
```

Rule (enforced by `tests/test_core_independence.py`): a core package imports only itself and `core.common`.
Each declares what it needs as a Protocol; `orchestrator/adapters.py` supplies the implementations.

## Data model

- **Structure** — `Block` stream → `StructureBuilder` → nodes DOCUMENT / SECTION / PARAGRAPH / SENTENCE / TABLE / FIGURE /
  EQUATION / CODE; edges CONTAINS / NEXT / REFERS_TO / CAPTIONS. Formats: pdf, html, docx, txt, md, pptx. Stored as JSON,
  and the structure store holds **structure only** (no body text). Text lives in the content store; embeddings in Chroma.
- **Time axis** — the reading-order `ordinal`. Retrieval is filtered by `ordinal <= position`; every concept, relation and
  definition is stamped with the ordinal where it was introduced; prerequisite edges only run earlier → later.
- **Knowledge graph** — starts empty; one LLM call per section sees the concepts known so far and returns a delta.
  Names are canonicalised (normalised text + embeddings, threshold 0.9). Deltas are kept, so the graph can be rewound to
  any section and replayed; a failed step is recorded and retried in order. Document status: `ready | partial | failed | none`.
- **Learner** — per (user, document) belief per concept (BKT: observe, decay, exposure). Evidence: quiz answers, questions
  asked, feedback, reading, self-report. Beliefs spread over graph neighbours; forgetting is applied at read time.
  A diagnostic quiz picks concepts by importance × uncertainty and recalibrates the starting level.
- **Adaptation** — `tutor/policy.py` turns the beliefs around the current section into a level plus rules: assume concepts
  ≥ 0.70, define concepts < 0.40, cap new terms per level. Recommended resources are earlier sections of the same document
  (`doc://{doc}#{section}`).

## Service

- Registration is a background job (single worker, bounded queue, dedupe by sha256, progress 0–5 % parse, 5–35 % index,
  35–100 % knowledge). Jobs are persisted; jobs interrupted by a restart are reported failed.
- Privacy: per-user via `X-User-Id`; optional bearer API key. Another user's documents are `not_found`.
- The SDK's operation table drives both protocols (`protocol="rest"|"rpc"`); sync and async clients expose the same methods.

## Tests

`tests/` (backend, real `System` + scripted LLM), `sdk/tests`, `ui/tests`. `tests/test_sdk_contract.py` runs the SDK against
the real app over both protocols.

## Configuration (one source per setting)

Every default value is written once, in a packaged YAML file; code never repeats it (`tests/test_config_layers.py` checks this).

| Config | Defaults (packaged) | Your overrides | Environment |
|---|---|---|---|
| LLM | `src/doc_explainer/config/defaults/llm_config.yaml` | `~/.doc_explainer/config/llm_config.yaml` | `DOC_EXPLAINER_LLM_<KEY>` |
| Backend | `.../defaults/backend_config.yaml` | `.../backend_config.yaml` | `DOC_EXPLAINER_<SECTION>_<KEY>` |
| UI | `ui/src/doc_explainer_ui/defaults/ui_config.yaml` | `.../ui_config.yaml` | `DOC_EXPLAINER_UI_<SECTION>_<KEY>` |
| Server process | `ApiSettings` | - | `DOC_EXPLAINER_HOST/PORT/API_KEY/DATA_DIR/...` |

Order: packaged defaults, then your file (only the keys you changed), then environment. Files are never copied whole, so a better
default in a new release reaches everyone; the first-run file is a commented copy to edit. `save()` writes only differences.
Provider facts (model, endpoint, key variable) are in `providers:` of `llm_config.yaml`; `llm.model: null` means that provider's
default. API keys are only ever read from the environment / `.env` (`AIROUTER_API_KEY`, `OPENROUTER_API_KEY`, ...).
