# 📘 Project: Doc Explainer - A Context-Aware, Knowledge-Adaptive Document Tutor Using Large Language Models

An **AI-augmented document viewer** that:

1. Explains **selected text** in context
2. Maintains **full-document understanding**
3. Builds and updates a **user knowledge model**
4. Adapts explanation **depth dynamically**
5. Supports **voice interaction**
6. Recommends **prerequisites or deeper material**


## 1. Problem 


Project focuses on modular development of an adaptive, document-centric intelligent tutoring system. By combining scalable document retrieval, flexible user modeling, and multimodal interaction within a unified architecture, the system enables rigorous experimentation while remaining deployable in real-world educational settings.

---



Given a:

1. **Document Tree**

   * Chapter → Section → Paragraph → Sentence
2. **User stops at a paragraph**

   * Input sentence → Get Paragraph ID → Section ID
   * Extract user selected texts
3. **Dependencies**

   * The paragraph depends on concepts introduced earlier
4. **User Knowledge**

   * User current knowlege (Which chapters/concepts the user knows)
   * Possibly *how well* they know them
5. **Goal**

   * Explain the paragraph:

     * Using only concepts the user already knows 
     * Re-explaining missing prerequisites (User choise)
     * At the *right depth* for that user

It’s a **Personalized, dependency-aware, concept-grounded explanation generation**

---

## Key Functionalities

### 🔹 1. Selected Text Explanation (Core Feature)

**Input:**

* Selected text span
* Surrounding section
* Entire document summary
* User knowledge state



This is **contextual explanation**, not generic paraphrasing.

---

### 🔹 2. Full-Document Context Awareness

Use **hierarchical memory**:

```
Document
├── Section summaries
│     ├── Paragraph embeddings
│           ├── Sentence embeddings
│
├── Concept map
│     ├── Concept → Sections
|
├── Knowlege map
```

**Methods:**

* Recursive chunking
* Section-level embeddings
* Concept extraction using LLM + NER

This enables:

* Cross-references (“As defined earlier in Section 2…”)
* Avoids hallucination
DocExplainer: Complete Implementation Guide



## 3. How This Differs From Existing Systems

| System       | Limitation                     |
| ------------ | ------------------------------ |
| ChatGPT      | No persistent user knowledge   |
| Kindle hints | Static, non-personal           |
| Khan Academy | Linear curriculum              |
| Copilot      | Task-oriented, not pedagogical |

This system is:

> **Dynamic + personalized + dependency-aware**


## 4. Project structure
```python
# File Structure:
# doc_explainer/
# ├── core/
# │   ├── document
# │   │     ├── document_processor.py    # Document parsing & chunking
# │   │     ├── document_processor.py    # Document parsing & chunking
# │   ├── knowlege_modeling
# │   │     ├── knowledge_graph.py       # knowlege graph implementation
# │   │     ├── knowledge_tracing.py     # Concept extraction & graph building for the document
# │   │     ├── user_model.py       # User knowlege state modeling / Bayesian knowledge tracing
# │   ├── explanation_engine           
# │   │     ├── adaptive_explainer.py   # Context-aware explanations
# │   │         └── resource_recomender.py   # Resource recomendation engine for new topics
# |   ├── interraction 
# │   │     ├──  voice_interface.py      # ASR + TTS
# |   ├── evaluation 
# |   |     ├── knowledge_evaluator.py   # Quiz generation
# |   |     ├── Explanation_evaluator.py # Explanation quality
# │   │     └── 
# ├── models/
# │   ├── embedding_model.py      # Custom embedding fine-tuning
# │   └── concept_extractor.py    # NER + LLM concept extraction
# ├── memory/
# │   ├── hierarchical_memory.py   # Document context memory
# │   └── vector_store.py         # FAISS + ChromaDB
# ├── ui/
# |   ├── gui
# │   │     ├── ...py   
# │   ├── web_app.py              # Streamlit/FastAPI interface
# │   |     ├── ...py  
# └── 
```

---

PART 1: DOCUMENT PROCESSOR 
PART 2: HIERARCHICAL MEMORY (INCLUDING CACHE)
PART 3: CONCEPT EXTRACTION & KNOWLEDGE GRAPH
PART 4: BAYESIAN KNOWLEDGE TRACING & USER MODELING
PART 5: ADAPTIVE EXPLAINER & CONTEXT-AWARE GENERATION
PART 6: VOICE INTERFACE & ASR/TTS
PART 7: MAIN APPLICATION & INTEGRATION


COMPONENTS TO IMPLEMENT:
## Core module
    * document  
        - Document caching
        - Document processing
    * memory
        - long_term_memory
        - knowlege caching
    * knowlege modelling
        - concept extrcaction 
        - knowlege graph / concept graph
        - user / knowlege model
    * explanation engine
        - adaptive explainer
        - resource recomender
        - voice interface
    * evaluation
        - knowledge_evaluator
    * ui
        - gui
        - web app



---



### Tasks to be completed
* PDF viewer (React / Electron)
* Basic voice input/output
* Text selection → sidebar explanation
* Concept graph extraction
* Knowledge mastery estimation
* Simple user profile (known / unknown)
* Adaptive explanation depth
* Context-aware RAG
* Prerequisite suggestion (recomendation)
* Personalized curriculum generation
* Quiz-based feedback loop
* Reinforcement learning for tutoring policy
* Multi-doc knowledge transfer
* Manual concept graph
* Paragraph → concept tagging
* Rule-based explanation prompt
* LLM-based explanation
* Automatic concept extraction
* Embedding-based concept linking
* Persistent user knowledge store
* Explanation quality feedback loop
* RL-based explanation depth tuning
* Multimodal (equations + diagrams)

---


