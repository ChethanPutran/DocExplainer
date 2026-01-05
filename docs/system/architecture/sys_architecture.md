
## System Architecture (High-Level)

```
Frontend (Web / Desktop)
│
├── Document Viewer (PDF / Markdown / HTML)
│     ├── Text Selection Listener
│     ├── Section + Position Tracking
│
├── Sidebar AI Tutor
│     ├── Explanation Panel
│     ├── Voice Output (TTS)
│     ├── Voice Input (ASR)
│     ├── Doubt / Question Interface
│
Backend
│
├── Document Understanding Engine
│     ├── Hierarchical Chunking
│     ├── Section Graph
│     ├── Embedding Store (Doc Memory)
│
├── User Knowledge Model
│     ├── Concept Graph
│     ├── Mastery Estimation
│     ├── Interaction History
│
├── Context-Aware Reasoning Engine
│     ├── Selected Text Context
│     ├── Global Doc Context
│     ├── User Knowledge Context
│
├── Recommendation Engine
│     ├── Prerequisite Detector
│     ├── Depth Estimator
│     ├── External Material Retriever
│
└── LLM Orchestration Layer
```
