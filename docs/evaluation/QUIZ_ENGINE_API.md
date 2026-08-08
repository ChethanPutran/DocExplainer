# Quiz Engine API Reference

## QuizEngine

The main orchestrator for quiz sessions with adaptive difficulty, immediate feedback, and mastery tracking.

### Initialization

```python
QuizEngine(
    quiz_generator: Optional[QuizGenerator] = None,
    mastery_estimator: Optional[MasteryEstimator] = None,
    difficulty_config: Optional[DifficultyAdjustmentConfig] = None
)
```

### Methods

#### `create_adaptive_quiz()`

Creates an adaptive quiz based on current mastery levels.

```python
def create_adaptive_quiz(
    user_id: str,
    concepts: List[str],
    knowledge_states: Dict[str, KnowledgeState],
    num_questions: int = 5,
) -> Tuple[Quiz, QuizSession]
```

**Parameters:**
- `user_id`: ID of the user taking the quiz
- `concepts`: List of concept names to assess
- `knowledge_states`: Current knowledge states for each concept
- `num_questions`: Number of questions to generate (default: 5)

**Returns:**
- Tuple of (Quiz object, QuizSession)

**Example:**
```python
quiz, session = engine.create_adaptive_quiz(
    user_id="user123",
    concepts=["Python", "Functions"],
    knowledge_states=states,
    num_questions=5
)
```

---

#### `submit_response()`

Submits a response to a quiz question and generates feedback.

```python
def submit_response(
    session_id: str,
    question_id: str,
    user_answer: str,
    response_time_seconds: Optional[float] = None,
) -> QuizResponse
```

**Parameters:**
- `session_id`: ID of the quiz session
- `question_id`: ID of the question being answered
- `user_answer`: User's response text
- `response_time_seconds`: Time taken to answer (optional)

**Returns:**
- QuizResponse object with feedback

**Raises:**
- `ValueError`: If session or question not found

**Example:**
```python
response = engine.submit_response(
    session_id="sess1",
    question_id="q1",
    user_answer="Python",
    response_time_seconds=8.5
)
print(response.feedback.explanation)
```

---

#### `get_next_question()`

Retrieves the next question in the session with adaptive difficulty.

```python
def get_next_question(
    session_id: str,
    knowledge_states: Dict[str, KnowledgeState],
) -> Optional[Question]
```

**Parameters:**
- `session_id`: ID of the quiz session
- `knowledge_states`: Current knowledge states

**Returns:**
- Next Question or None if all questions answered

**Example:**
```python
next_q = engine.get_next_question(session_id="sess1", knowledge_states=states)
if next_q:
    print(f"Next question: {next_q.text}")
```

---

#### `get_session_progress()`

Gets current progress in a quiz session.

```python
def get_session_progress(session_id: str) -> Dict[str, Any]
```

**Parameters:**
- `session_id`: ID of the session

**Returns:**
- Dictionary with progress information:
  - `session_id`: Session ID
  - `total_questions`: Total questions in quiz
  - `answered_questions`: Number answered so far
  - `remaining_questions`: Number not yet answered
  - `correct_count`: Number of correct answers
  - `score_percentage`: Current score as percentage
  - `is_completed`: Whether session is completed
  - `started_at`: Session start timestamp
  - `elapsed_seconds`: Time elapsed in seconds

**Example:**
```python
progress = engine.get_session_progress("sess1")
print(f"Score: {progress['score_percentage']}%")
print(f"Progress: {progress['answered_questions']}/{progress['total_questions']}")
```

---

#### `complete_session()`

Completes a quiz session and updates all mastery states.

```python
def complete_session(
    session_id: str,
    knowledge_states: Dict[str, KnowledgeState],
) -> QuizSession
```

**Parameters:**
- `session_id`: ID of the session to complete
- `knowledge_states`: Knowledge states to update

**Returns:**
- Completed QuizSession with final statistics and mastery updates

**Modifies:**
- Updates knowledge_states dict in-place
- Sets mastery_updates on session

**Example:**
```python
completed = engine.complete_session("sess1", knowledge_states)
print(f"Final score: {completed.score_percentage}%")
for concept, change in completed.mastery_updates.items():
    print(f"{concept}: {change:+.3f}")
```

---

#### `export_session()`

Exports a quiz session as a dictionary.

```python
def export_session(session_id: str) -> Dict[str, Any]
```

**Parameters:**
- `session_id`: ID of the session

**Returns:**
- Dictionary representation of session

---

#### `import_session()`

Imports a quiz session from a dictionary.

```python
def import_session(session_data: Dict[str, Any]) -> QuizSession
```

**Parameters:**
- `session_data`: Dictionary with session data

**Returns:**
- QuizSession object

---

## AdaptiveDifficultyManager

Manages adaptive difficulty adjustment based on mastery level and response correctness.

### Methods

#### `get_difficulty_for_mastery()`

Returns difficulty level for a given mastery probability.

```python
def get_difficulty_for_mastery(mastery_level: float) -> DifficultyLevel
```

**Parameters:**
- `mastery_level`: Mastery probability (0.0-1.0)

**Returns:**
- DifficultyLevel (BEGINNER, INTERMEDIATE, or ADVANCED)

**Mapping:**
- mastery < 0.3 → BEGINNER
- 0.3 ≤ mastery < 0.7 → INTERMEDIATE
- mastery ≥ 0.7 → ADVANCED

---

#### `adjust_difficulty()`

Adjusts difficulty based on response correctness streak.

```python
def adjust_difficulty(
    concept: str,
    is_correct: bool
) -> Optional[DifficultyLevel]
```

**Parameters:**
- `concept`: Name of the concept
- `is_correct`: Whether the response was correct

**Returns:**
- New DifficultyLevel if adjustment needed, None otherwise

---

## FeedbackGenerator

Generates immediate, explanatory feedback with learning resources.

### Methods

#### `generate_feedback()`

Generates feedback for a quiz response.

```python
def generate_feedback(
    question: Question,
    user_answer: str,
    is_correct: bool,
    mastery_level: float = 0.5,
) -> QuizFeedback
```

**Parameters:**
- `question`: The question being answered
- `user_answer`: User's response
- `is_correct`: Whether response is correct
- `mastery_level`: Current mastery level (0-1)

**Returns:**
- QuizFeedback object

**Example:**
```python
feedback = generator.generate_feedback(
    question=q1,
    user_answer="Python",
    is_correct=True,
    mastery_level=0.75
)
```

---

## MasteryUpdater

Updates knowledge states based on quiz responses using BKT.

### Methods

#### `update_mastery_from_response()`

Updates knowledge state from a response.

```python
def update_mastery_from_response(
    knowledge_state: KnowledgeState,
    is_correct: bool,
    response_time_seconds: Optional[float] = None,
) -> Tuple[KnowledgeState, float]
```

**Parameters:**
- `knowledge_state`: Current knowledge state
- `is_correct`: Whether response was correct
- `response_time_seconds`: Time taken to answer

**Returns:**
- Tuple of (updated_knowledge_state, confidence_change)

**Example:**
```python
updated, change = updater.update_mastery_from_response(
    state,
    is_correct=True,
    response_time_seconds=5.0
)
print(f"Mastery change: {change:+.3f}")
```

---

## QuizRepository

Repository for quiz data persistence.

### Initialization

```python
QuizRepository(storage_path: Optional[str] = None)
```

**Parameters:**
- `storage_path`: Optional path for file-based storage. If None, uses in-memory storage.

### Methods

#### `save_session()`

Saves a quiz session.

```python
def save_session(session: QuizSession) -> bool
```

**Returns:**
- True if successful, False otherwise

---

#### `load_session()`

Loads a quiz session by ID.

```python
def load_session(session_id: str) -> Optional[QuizSession]
```

**Returns:**
- QuizSession or None if not found

---

#### `get_all_user_sessions()`

Gets all sessions for a user.

```python
def get_all_user_sessions(user_id: str) -> List[QuizSession]
```

**Returns:**
- List of QuizSession objects

---

#### `save_response()`

Saves a response for a session.

```python
def save_response(session_id: str, response: QuizResponse) -> bool
```

**Returns:**
- True if successful

---

#### `get_responses_for_session()`

Gets all responses for a session.

```python
def get_responses_for_session(session_id: str) -> List[QuizResponse]
```

**Returns:**
- List of QuizResponse objects

---

#### `track_question_performance()`

Tracks performance metrics for a question.

```python
def track_question_performance(
    question_id: str,
    concept: str,
    is_correct: bool,
    response_time_seconds: Optional[float] = None,
) -> None
```

---

#### `get_question_performance()`

Gets performance metrics for a question.

```python
def get_question_performance(question_id: str) -> Optional[Dict[str, Any]]
```

**Returns:**
- Performance dict with:
  - `attempts`: Total attempts
  - `correct`: Correct attempts
  - `accuracy`: Accuracy percentage
  - `avg_time_seconds`: Average response time

---

#### `get_concept_performance()`

Gets aggregated performance for a concept.

```python
def get_concept_performance(concept: str) -> Dict[str, Any]
```

**Returns:**
- Performance dict with:
  - `total_questions`: Number of unique questions
  - `total_attempts`: Total attempts on concept
  - `accuracy`: Overall accuracy
  - `avg_time_seconds`: Average response time

---

## Enums

### QuestionType
```python
class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    MATCHING = "matching"
    ESSAY = "essay"
```

### DifficultyLevel
```python
class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    ADAPTIVE = "adaptive"
```

### ResponseCorrectness
```python
class ResponseCorrectness(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    NOT_ATTEMPTED = "not_attempted"
```

---

## Error Handling

### Common Exceptions

```python
# Session not found
try:
    engine.get_session_progress("invalid_id")
except ValueError as e:
    print(f"Error: {e}")  # Quiz session invalid_id not found

# Question not found in session
try:
    engine.submit_response("sess1", "invalid_q", "answer")
except ValueError as e:
    print(f"Error: {e}")  # Question invalid_q not found in session
```

---

## Performance Considerations

1. **Session Storage**: Use file-based storage for production
2. **Question Caching**: Reuse QuizGenerator across requests
3. **Batch Updates**: Update mastery only on session completion
4. **Index Performance**: For large repositories, implement database indexing

---

## Thread Safety

The QuizEngine is not thread-safe by default. For multi-threaded applications:
- Use thread-local storage for active_sessions
- Implement locking around session operations
- Consider using async/await patterns
