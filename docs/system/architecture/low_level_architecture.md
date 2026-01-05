
---

## 2️⃣ Implementation-Level Architecture (Engineering View)

```
┌─────────────── Frontend (React / Next.js) ───────────────┐
│                                                           │
│  PDF.js Viewer ──► Text Selection Events                  │
│         │                                                 │
│         ├──► Sidebar UI (Explanations / Rec)              │
│         │                                                 │
│         └──► Voice Input ──► ASR (Whisper)                │
│                                                           │
└───────────────┬───────────────────────────────────────────┘
                │ REST / WebSocket
                ▼
┌──────────────────── Backend (FastAPI) ───────────────────┐
│                                                           │
│  Interaction Controller                                   │
│  • Session Manager                                        │
│  • Multimodal Input Handler                               │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Context Builder                                      │ │
│  │ • Selected Span                                      │ │
│  │ • Section Context                                    │ │
│  │ • Retrieved Doc Chunks (FAISS/Milvus)                │ │
│  │ • User Knowledge Summary                              │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Knowledge State Estimator                             │ │
│  │ • Mastery Update Equations                            │ │
│  │ • Concept Graph Propagation                           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ LLM Orchestration Layer                               │ │
│  │ • Prompt Assembly                                     │ │
│  │ • Explanation Depth Control                           │ │
│  │ • LLM Inference (LLaMA / GPT / Mistral)               │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Recommendation Engine                                 │ │
│  │ • Prerequisite Detection                              │ │
│  │ • Advanced Content Retrieval                          │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌──────────────────── Data & Memory Layer ──────────────────┐
│                                                           │
│  Document Store                                           │
│  • Raw Documents                                          │
│  • Section Metadata                                       │
│                                                           │
│  Vector Store (FAISS / Milvus)                             │
│  • Paragraph / Section Embeddings                         │
│                                                           │
│  User Model Store (SQLite / PostgreSQL)                    │
│  • Interaction Logs                                       │
│  • Knowledge States                                       │
│                                                           │
│  Concept Graph (NetworkX / Neo4j)                          │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---
