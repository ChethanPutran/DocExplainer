# Quiz & Feedback Engine

## Overview

The Quiz & Feedback Engine is a comprehensive system for generating adaptive quizzes, providing immediate feedback with explanations, and tracking mastery improvements based on user responses. It integrates Bayesian Knowledge Tracing (BKT) for intelligent knowledge state management.

## Architecture

### Core Components

#### 1. **QuizEngine** (Main Orchestrator)
- Manages quiz lifecycle: creation, session management, response evaluation
- Coordinates between question generation, feedback generation, and mastery updates
- Maintains active quiz sessions with state tracking

#### 2. **AdaptiveDifficultyManager**
- Adjusts question difficulty based on learner mastery level
- Maps mastery to difficulty levels:
  - **Easy** (< 0.3 mastery): Basic recall questions
  - **Medium** (0.3-0.7 mastery): Application and synthesis questions
  - **Hard** (> 0.7 mastery): Analysis and evaluation questions
- Tracks consecutive correct/incorrect responses for dynamic difficulty adjustment

#### 3. **FeedbackGenerator**
- Produces immediate, explanatory feedback for every response
- Provides:
  - Correct/incorrect indicator
  - Detailed explanation of correct answer
  - Hints for incorrect responses
  - Related document sections
  - Personalized next steps based on mastery level

#### 4. **MasteryUpdater**
- Updates knowledge states using BKT algorithm
- Integrates with MasteryEstimator to calculate p_knowledge changes
- Tracks confidence changes and updates interaction history
- Triggers curriculum adjustments based on performance

#### 5. **QuizGenerator**
- Generates multi-format questions:
  - Multiple Choice (3-5 options)
  - True/False
  - Fill-in-the-blank
  - Short answer
- Supports adaptive question generation based on concepts and difficulty

#### 6. **QuizRepository**
- Persists quiz sessions and responses
- Supports both in-memory and file-based storage
- Tracks question performance metrics
- Aggregates performance by concept

## Features

### Multi-Format Question Generation

#### Question Types
- **Multiple Choice**: 3-5 options with explanations
- **True/False**: Binary choice with reasoning
- **Fill-in-the-blank**: Text matching with fuzzy comparison
- **Short Answer**: LLM-based scoring with similarity matching

#### Example: Creating a Quiz

```python
from src.core.evaluation.quiz_engine import QuizEngine
from src.core.evaluation.models.enums import DifficultyLevel

# Initialize quiz engine
engine = QuizEngine()

# Create adaptive quiz based on mastery levels
quiz, session = engine.create_adaptive_quiz(
    user_id="user123",
    concepts=["Python Basics", "Functions", "Classes"],
    knowledge_states=user_knowledge_states,
    num_questions=5
)
```

### Adaptive Difficulty Based on Mastery

The engine automatically adjusts question difficulty based on learner performance:

```python
# Get next question with adaptive difficulty
next_question = engine.get_next_question(
    session_id=session.id,
    knowledge_states=knowledge_states
)
```

**Difficulty Adjustment Rules:**
- Mastery < 0.3 → BEGINNER (recall questions)
- Mastery 0.3-0.7 → INTERMEDIATE (application questions)
- Mastery > 0.7 → ADVANCED (analysis questions)
- 2 consecutive correct → Increase difficulty
- 2 consecutive incorrect → Decrease difficulty

### Immediate Feedback with Explanations

Every response receives immediate, personalized feedback:

```python
# Submit response and get feedback
response = engine.submit_response(
    session_id=session.id,
    question_id=question.id,
    user_answer="Python",
    response_time_seconds=8.5
)

feedback = response.feedback
# Includes:
# - is_correct: Boolean
# - explanation: Detailed explanation
# - hint: For incorrect responses
# - related_sections: Related topics
# - confidence_score: Confidence in assessment
# - next_step: Recommendation for next action
```

### Mastery Tracking and Updates

Uses BKT to update learner knowledge states:

```python
# Complete session and update all masteries
completed_session = engine.complete_session(
    session_id=session.id,
    knowledge_states=knowledge_states
)

# mastery_updates contains change for each concept
for concept, change in completed_session.mastery_updates.items():
    print(f"{concept}: {change:+.3f}")
```

## Data Models

### QuizQuestion
```python
{
    "id": "q1",
    "text": "What is Python?",
    "type": "multiple_choice",
    "difficulty": "beginner",
    "concept": "Python Basics",
    "options": [...],
    "correct_answer": "A programming language",
    "explanation": "Detailed explanation...",
    "hints": ["Think about programming", "It's used for AI"],
    "tags": ["programming", "basics"],
    "created_at": "2024-05-15T10:00:00Z"
}
```

### QuizResponse
```python
{
    "question_id": "q1",
    "user_answer": "A programming language",
    "is_correct": true,
    "timestamp": "2024-05-15T10:02:00Z",
    "response_time_seconds": 8.5,
    "feedback": {...}
}
```

### QuizFeedback
```python
{
    "is_correct": true,
    "explanation": "Correct! Python is indeed a high-level programming...",
    "hint": null,
    "related_sections": ["Python Basics", "beginner"],
    "confidence_score": 0.95,
    "next_step": "Try a harder question"
}
```

### QuizSession
```python
{
    "id": "session1",
    "quiz_id": "quiz1",
    "user_id": "user123",
    "questions": [...],
    "responses": [...],
    "started_at": "2024-05-15T10:00:00Z",
    "completed_at": null,
    "session_stats": {
        "total_questions": 5,
        "answered_questions": 3,
        "correct_count": 2,
        "score_percentage": 40.0
    },
    "mastery_updates": {
        "Python Basics": 0.05,
        "Functions": -0.02
    }
}
```

## Usage Examples

### Example 1: Complete Quiz Flow

```python
from src.core.evaluation.quiz_engine import QuizEngine
from src.core.evaluation.models.enums import DifficultyLevel
from src.core.evaluation.repository.quiz_repository import QuizRepository

# Initialize components
engine = QuizEngine()
repository = QuizRepository(storage_path="./quiz_data")

# Create adaptive quiz
quiz, session = engine.create_adaptive_quiz(
    user_id="user123",
    concepts=["Python Basics", "Functions"],
    knowledge_states=knowledge_states,
    num_questions=5
)

# Take the quiz
for question in session.questions:
    print(f"Question: {question.text}")
    user_answer = input("Your answer: ")
    
    # Submit response
    response = engine.submit_response(
        session_id=session.id,
        question_id=question.id,
        user_answer=user_answer,
        response_time_seconds=time_spent
    )
    
    # Display feedback
    print(f"Feedback: {response.feedback.explanation}")
    
    # Save response
    repository.save_response(session.id, response)

# Complete session
completed_session = engine.complete_session(
    session_id=session.id,
    knowledge_states=knowledge_states
)

# Save final session
repository.save_session(completed_session)

# Print results
print(f"Score: {completed_session.score_percentage}%")
print(f"Mastery updates: {completed_session.mastery_updates}")
```

### Example 2: Targeting Knowledge Gaps

```python
from src.core.evaluation.generators.quiz_generator import QuizGenerator

# Generate quiz for knowledge gaps
generator = QuizGenerator()
quiz = generator.generate_quiz_from_knowledge_gaps(
    unknown_concepts=["Decorators", "Generators"],
    known_concepts=["Functions", "Classes"],
    num_questions=5
)
```

### Example 3: Mastery Assessment

```python
# Generate mastery quiz
quiz = generator.generate_mastery_quiz(
    concept="Functions",
    mastery_level=0.65,  # 65% mastery
    num_questions=3
)
```

## Integration Points

### With MasteryEstimator
- Calls `update_from_response()` to calculate new p_knowledge
- Passes InteractionResponse with correctness and timing
- Updates BKT parameters based on learning rate

### With AdaptiveExplainer
- Uses for generating detailed feedback explanations
- Leverages LLM for complex question types
- Provides context from related documents

### With Knowledge Graph
- Retrieves related concepts for feedback
- Finds prerequisite relationships
- Suggests follow-up topics

## Configuration

### DifficultyAdjustmentConfig

```python
from src.core.evaluation.quiz_engine import DifficultyAdjustmentConfig

config = DifficultyAdjustmentConfig(
    easy_threshold=0.3,           # Mastery threshold for easy/medium
    medium_threshold=0.7,          # Mastery threshold for medium/hard
    consecutive_correct_for_increase=2,   # Consecutive correct to increase
    consecutive_incorrect_for_decrease=2  # Consecutive incorrect to decrease
)

engine = QuizEngine(difficulty_config=config)
```

## Performance Metrics

### Quiz Metrics
- **Score Percentage**: Correct answers / total questions
- **Response Time**: Average time per question
- **Difficulty Progression**: Questions increased/decreased in difficulty
- **Confidence Score**: Confidence in each assessment

### Mastery Metrics
- **p_knowledge**: Probability of knowledge (0-1)
- **n_attempts**: Total attempts on concept
- **n_correct**: Correct attempts
- **Accuracy**: n_correct / n_attempts
- **Learning Rate**: Rate of p_knowledge improvement

## Storage

### In-Memory Storage
```python
repository = QuizRepository()  # Uses in-memory storage
```

### File-Based Storage
```python
repository = QuizRepository(storage_path="./quiz_sessions")
# Sessions saved as: session_{id}.json
# Responses saved as: responses_{id}.json
```

### Data Export
```python
# Export all data
all_data = repository.export_all_data()
# Returns: {'sessions': {...}, 'responses': {...}, 'question_performance': {...}}
```

## Testing

All components are thoroughly tested:

```bash
pytest src/core/evaluation/tests/test_quiz_engine.py -v
```

**Test Coverage:**
- Adaptive difficulty management (7 tests)
- Feedback generation (4 tests)
- Mastery updates (3 tests)
- Quiz engine functionality (9 tests)
- Quiz repository (5 tests)
- Session statistics (2 tests)

## Best Practices

1. **Session Management**
   - Create one session per quiz attempt
   - Complete session before starting new one
   - Store sessions for analytics

2. **Difficulty Adaptation**
   - Monitor consecutive correct/incorrect patterns
   - Adjust difficulties gradually
   - Provide harder questions as mastery increases

3. **Feedback Quality**
   - Always provide explanations with feedback
   - Include hints for incorrect responses
   - Link to related topics

4. **Mastery Tracking**
   - Update knowledge states after each response
   - Adjust learning parameters based on performance
   - Track mastery history over time

## Future Enhancements

- Real-time adaptive sequencing
- Multi-modal question types (images, video)
- Peer comparison analytics
- Spaced repetition optimization
- Collaborative quiz modes
- Advanced LLM-based scoring
