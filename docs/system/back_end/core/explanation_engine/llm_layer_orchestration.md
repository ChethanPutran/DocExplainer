
## 5. Large Language Model Layer

### Language Models

* **Open-source**: LLaMA, Mistral, Qwen
* **API-based (prototyping)**: GPT-class models
  *Justification*: Open-source models support reproducibility; APIs accelerate early development.

### Prompt Orchestration

* **Custom Orchestration Layer / LangChain**
  *Justification*: Enables structured prompts incorporating document context, user knowledge summaries, and explanation depth control.

---


## 8. Asynchronous Processing and Orchestration

### Task Queues

* **Celery with Redis**
  *Justification*: Handles long-running tasks such as embedding computation and explanation generation.

### Caching

* **Redis**
  *Justification*: Reduces latency for repeated context retrieval and explanation requests.

---