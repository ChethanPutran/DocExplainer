## High-Level System Architecture (Conceptual)

```
┌────────────────────────────┐
│        User Interface      │
│────────────────────────────│
│  Document Viewer (PDF/MD)  │
│  • Text Selection          │
│  • Highlighting            │
│                            │
│  AI Tutor Sidebar          │
│  • Explanation Display     │
│  • Recommendations         │
│  • Voice Output (TTS)      │
│                            │
│  Voice Input (ASR)         │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│   Interaction Orchestrator │
│────────────────────────────│
│ • Event Normalization      │
│ • Session Context          │
│ • Multimodal Fusion        │
└──────────────┬─────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│        Context Construction Engine            │
│──────────────────────────────────────────────│
│ • Selected Text Span                          │
│ • Local Section Context                      │
│ • Global Document Context (Retrieval)        │
│ • User Knowledge Summary                     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│   Adaptive Reasoning & Explanation Engine    │
│──────────────────────────────────────────────│
│ • Explanation Depth Selection                │
│ • Prompt Conditioning                       │
│ • LLM-based Explanation Generation           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Recommendation & Pedagogy Engine             │
│──────────────────────────────────────────────│
│ • Prerequisite Detection                    │
│ • Advanced Material Suggestion               │
│ • Learning Path Guidance                    │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│        User Knowledge Model                  │
│──────────────────────────────────────────────│
│ • Concept Mastery Estimation                │
│ • Knowledge State Update                    │
│ • Forgetting & Propagation                  │
└──────────────────────────────────────────────┘
```