Once the document texts have be converted into Document Tree structure. We can use it to develop knowlege graph. It requires **Three representations**:

### (A) Document → Concept Mapping

Each paragraph is mapped to:

* Primary concepts
* Secondary (implicit) concepts

Example:

```
Paragraph P42:
  primary: Backpropagation
  secondary: Chain Rule, Partial Derivatives, Computational Graph
```

---
### (B) Concept Graph (Knowledge Graph)

* Nodes = concepts
* Edges = *depends-on*, *extends*, *uses*

Example:

```
Gradient Descent
 ├── depends on → Derivative
 ├── depends on → Loss Function
 └── extends → Optimization
```
### Concept Graph Storage

* **NetworkX (research)** / **Neo4j (optional)**
  *Justification*: Graph-based representation of concept dependencies and prerequisite relations.
---



### (C) User Knowledge Model

User Knowledge Representation (Important)

```json
{
  "concept": "Chain Rule",
  "subject":"Calculus",
  "mastery": 0.42,
  "level":"Beginner / intermediate / advanced mastery",
  "last_seen": "2026-01-01",
  "time_spent": "20hr",
  "quiz_response": "8/10",
  "questions_asked": "list of questions",
  "explanation_depth_requested": "...",
  "source": "Chapter 3",
}
```
**Modeling approaches:**

* Bayesian Knowledge Tracing
* Item Response Theory (advanced)
* LLM-based mastery estimation (initial MVP)

### Kowledge Modeling and Storage
* **In-Memory Data Structures** for real-time updates
* **Persistent Storage**: SQLite / PostgreSQL
  *Justification*: Lightweight persistence for interaction logs and user profiles; easy migration to larger systems.






## Learning From Interaction (Very Important)

After explanation:

* Ask:

  * “Did this make sense?”
  * “Which part was unclear?”
* Update knowledge score based on user response:
This becomes **online user modeling**.

---
**Modules:**

1. **Difficulty Prediction**

   * Features: length, number of unknown concepts, embedding complexity
   * Supervised ML: Decision Tree / Random Forest / Logistic Regression
2. **User Mastery Prediction**

   * Features: time spent, explanations requested, quiz responses
   * Supervised ML: Random Forest / Gradient Boosting / Logistic Regression
3. **Adaptive Explanation Depth**

   * Map mastery → explanation style (beginner → advanced)
   * Example: mastery < 0.3 → intuition + analogies, mastery > 0.7 → formal definitions

**Deliverables:**

* Explanation depth adapts automatically
* Predictive user mastery updates after each interaction

---

