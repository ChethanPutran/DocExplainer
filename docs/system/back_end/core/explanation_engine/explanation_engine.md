## 3. Explanation Strategy (Key Logic)

When user selects a paragraph:

### Step 1: Identify Required Concepts

```text
Required = concepts(paragraph) 
          + prerequisites(concepts)
```

---

### Step 2: Compare with User Knowledge

```text
Missing = Required − Known(user)
```

---

### Step 3: Generate Explanation Plan

Explanation should:

1. Briefly recall known concepts
2. Re-explain missing prerequisites
3. Explain the paragraph using:

   * Familiar terminology
   * Familiar examples
   * Familiar abstractions

---

## 4. Explanation Pipeline (System Design)

### 1️⃣ Paragraph Understanding

Use NLP to extract:

* Key terms
* Equations
* Implicit assumptions

**Tech**:

* Transformer-based encoder
* Keyphrase extraction (NER + dependency parsing)
* Mathematical symbol parsing (if needed)

---

### 2️⃣ Concept Resolution

Map text → canonical concepts

Methods:

* Keyword → concept dictionary
* Embedding similarity (Sentence-BERT)
* Graph-based disambiguation

---

### 3️⃣ Prerequisite Expansion

Walk the concept graph backwards:

```python
def get_prereqs(concept):
    return graph.backward(concept)
```

---

### 4️⃣ User Knowledge Filtering

Tag concepts as:

* `known`
* `weak`
* `unknown`

---

### 5️⃣ Prompt / Explanation Planner

Construct a **structured explanation prompt**:

```
Explain concept X.

User already understands:
- A
- B

User does NOT understand:
- C
- D

Explain X using A and B.
First briefly explain C and D using A and B.
Avoid introducing new concepts.
```

This is **crucial**.

---

## 5. LLM Prompt Template (Example)

```text
You are a teaching assistant.

Target paragraph:
<paragraph text>

Concepts involved:
Primary: Backpropagation
Prerequisites: Chain Rule, Partial Derivatives, Computational Graph

User knowledge:
Known well: Derivative, Functions
Partially known: Chain Rule
Unknown: Computational Graph

Instructions:
1. Briefly refresh Chain Rule using derivatives
2. Explain Computational Graph from scratch
3. Explain the paragraph using only these concepts
4. Use intuitive examples
5. Do not introduce new math terms
```

This turns the LLM into a **controlled explainer**, not a hallucinator.



### 🔹 4. Adaptive Explanation Depth

| User Level   | Explanation Style               |
| ------------ | ------------------------------- |
| Beginner     | Intuition + analogies           |
| Intermediate | Math + examples                 |
| Advanced     | Formal definitions + references |

The system **chooses explanation mode dynamically**.

---
3. **Context-Aware Explanation**

   * LLM (OpenAI GPT / LLaMA)
   * Prompt: “Explain this text/code considering user mastery: X, known: Y, unknown: Z, in context of document”
### 🔹 5. Voice Explanation & Doubt Asking

**Pipeline:**

```
Voice → ASR → Query
Query + Context → LLM
Response → TTS
```

**Use cases:**

* “Explain this again more intuitively”
* “Why is this assumption valid?”
* “Give a counterexample”

---
