
KEY ML INNOVATIONS IN THE PROJECT:

1. Hierarchical Document Understanding:

· 4-level document parsing (document → sections → paragraphs → sentences)
· Recursive embeddings at each level
· Graph-based representation of document structure

2. Bayesian Knowledge Tracing:

· Probabilistic modeling of concept mastery
· Adaptive parameter updating based on interactions
· Confidence-aware knowledge estimates

3. Concept Graph Learning:

· Hybrid concept extraction (NER + patterns + LLM)
· Relationship extraction (prerequisites, co-occurrence)
· Graph-based prerequisite analysis

4. Adaptive Explanation Generation:

· Style adaptation (beginner/intermediate/advanced)
· Context-aware prompting
· Knowledge-gap bridging

5. Voice-Enabled Learning:

· Real-time ASR with Whisper fallback
· Natural language query understanding
· TTS for explanations

6. Curriculum Generation:

· Dynamic learning path creation
· Micro-lesson generation for gaps
· Resource recommendation


Built AI Knowledge Tutor, a context-aware, adaptive assistant for documents and code, using LLMs, embeddings, and hierarchical memory to generate personalized explanations.

Implemented user knowledge modeling (Bayesian Knowledge Tracing + classical ML) to track mastery and dynamically adapt explanation depth.

Designed recommendation engine and concept dependency graph to suggest prerequisites, exercises, and advanced content using KNN, clustering, and graph traversal.

Integrated diffusion-based embedding refinement and synthetic exercise generation to enhance user learning experience.

Developed real-time Copilot-style IDE/browser plugin with inline explanations, recommendations, and adaptive voice interaction.

Notice how 5 bullets can come from a single project — it demonstrates both technical depth and product sense.

## **DSA & SWE Applications**

* **Graphs:** Concept dependency, recommendation traversal
* **Heaps / Priority Queues:** Top-K recommendations
* **HashMaps:** Track per-user concept mastery
* **Trees / recursive traversal:** Section → paragraph → sentence embeddings
* **Vector Search:** FAISS / HNSW for top-K similarity retrieval
* **OOP:** Modular backend (User, ConceptGraph, RecommendationEngine, EmbeddingManager)


## **Tech Stack Summary**

| Layer             | Technology                                                                            |
| ----------------- | ------------------------------------------------------------------------------------- |
| Backend           | FastAPI, Uvicorn, GPU server                                                          |
| Frontend / Plugin | VS Code Extension API, Browser extension (JS/TS), Streamlit/Gradio for demo           |
| ML / DL           | LLMs (GPT / LLaMA), SentenceTransformer / CodeBERT, Diffusion (PyTorch / HuggingFace) |
| Classical ML      | Logistic Regression, Random Forest, KNN, KMeans, Gradient Boosting                    |
| Concept Graph     | networkx / Neo4j                                                                      |
| Database          | SQLite / Postgres for user models & logs                                              |
| Voice (Optional)  | Whisper ASR + TTS (pyttsx3 / FastSpeech)                                              |
| Vector Search     | FAISS / Milvus / Pinecone                                                             |

This is strong for placements/research:

* NLP parsing & representation learning
* Knowledge graphs
* Graph traversal & reasoning
* User modeling
* Retrieval-Augmented Generation (RAG)
* Prompt programming / planning
* Reinforcement learning (optional: optimize explanations)
