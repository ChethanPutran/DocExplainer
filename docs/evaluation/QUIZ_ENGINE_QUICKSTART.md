# Quiz Engine Quick Start Guide

## Installation

The Quiz Engine is part of the DocExplainer project. No additional dependencies are required beyond the main requirements.

## 5-Minute Quick Start

### 1. Basic Quiz Creation

```python
from src.core.evaluation.quiz_engine import QuizEngine
from src.core.evaluation.models.enums import DifficultyLevel

# Create the quiz engine
engine = QuizEngine()

# Create a simple quiz
quiz, session = engine.create_adaptive_quiz(
    user_id="student_001",
    concepts=["Python Basics", "Functions"],
    knowledge_states={
        "Python Basics": KnowledgeState(concept=..., p_knowledge=0.4),
        "Functions": KnowledgeState(concept=..., p_knowledge=0.6),
    },
    num_questions=5
)

print(f"Quiz created with {len(session.questions)} questions")
```

### 2. Taking a Quiz

```python
# Loop through questions
for question in session.questions:
    print(f"\n{question.text}")
    
    if question.type == QuestionType.MULTIPLE_CHOICE:
        for i, option in enumerate(question.options, 1):
            print(f"  {i}. {option.text}")
    
    # Get user answer
    user_input = input("Your answer: ")
    
    # Submit response
    response = engine.submit_response(
        session_id=session.id,
        question_id=question.id,
        user_answer=user_input
    )
    
    # Show feedback immediately
    print(f"\nFeedback: {response.feedback.explanation}")
    if response.feedback.hint:
        print(f"Hint: {response.feedback.hint}")
```

### 3. Getting Session Progress

```python
# Check progress at any time
progress = engine.get_session_progress(session.id)

print(f"Questions answered: {progress['answered_questions']}/{progress['total_questions']}")
print(f"Correct so far: {progress['correct_count']}")
print(f"Current score: {progress['score_percentage']}%")
```

### 4. Completing the Quiz

```python
# When all questions are answered
completed_session = engine.complete_session(
    session_id=session.id,
    knowledge_states=knowledge_states
)

# View results
print(f"\n=== Quiz Results ===")
print(f"Final Score: {completed_session.score_percentage}%")
print(f"Questions Correct: {sum(1 for r in completed_session.responses if r.is_correct)}/{len(completed_session.questions)}")

# View mastery improvements
print(f"\n=== Mastery Updates ===")
for concept, change in completed_session.mastery_updates.items():
    print(f"{concept}: {change:+.2%}")
```

## Complete Example: Full Quiz Application

```python
from src.core.evaluation.quiz_engine import QuizEngine
from src.core.evaluation.repository.quiz_repository import QuizRepository
from src.core.evaluation.models.enums import QuestionType, DifficultyLevel
from src.core.knowledge.models.concept import Concept
from src.core.user.models.knowledge_state import KnowledgeState

class SimpleQuizApp:
    def __init__(self):
        self.engine = QuizEngine()
        self.repo = QuizRepository()
    
    def create_sample_knowledge_states(self):
        """Create sample knowledge states for testing."""
        concepts = ["Python Basics", "Functions", "Loops"]
        states = {}
        
        for concept_name in concepts:
            concept = Concept(name=concept_name)
            states[concept_name] = KnowledgeState(
                concept=concept,
                p_knowledge=0.5,
                n_attempts=5,
                n_correct=3
            )
        
        return states
    
    def run_quiz(self, user_id: str, concepts: List[str], num_questions: int = 5):
        """Run a complete quiz session."""
        
        # Create knowledge states
        knowledge_states = self.create_sample_knowledge_states()
        
        # Create adaptive quiz
        quiz, session = self.engine.create_adaptive_quiz(
            user_id=user_id,
            concepts=concepts,
            knowledge_states=knowledge_states,
            num_questions=num_questions
        )
        
        print(f"\n{'='*50}")
        print(f"Quiz: {quiz.title}")
        print(f"{'='*50}\n")
        
        # Take the quiz
        for i, question in enumerate(session.questions, 1):
            print(f"Question {i}/{len(session.questions)}")
            print(f"Topic: {question.concept}")
            print(f"Difficulty: {question.difficulty.value}")
            print(f"\n{question.text}\n")
            
            # Display options if multiple choice
            if question.type == QuestionType.MULTIPLE_CHOICE:
                for j, option in enumerate(question.options, 1):
                    print(f"  {j}. {option.text}")
                print()
            
            # Get user answer
            user_answer = input("Your answer: ").strip()
            
            # Submit response
            response = self.engine.submit_response(
                session_id=session.id,
                question_id=question.id,
                user_answer=user_answer
            )
            
            # Show immediate feedback
            print(f"\n{'─'*50}")
            if response.is_correct:
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
            print(f"{response.feedback.explanation}")
            
            if response.feedback.hint:
                print(f"\nHint: {response.feedback.hint}")
            
            if response.feedback.next_step:
                print(f"Next: {response.feedback.next_step}")
            print(f"{'─'*50}\n")
            
            # Save response
            self.repo.save_response(session.id, response)
        
        # Complete the quiz
        completed = self.engine.complete_session(
            session_id=session.id,
            knowledge_states=knowledge_states
        )
        
        # Display final results
        self.display_results(completed, knowledge_states)
        
        # Save session
        self.repo.save_session(completed)
        
        return completed
    
    def display_results(self, session, knowledge_states):
        """Display quiz results and mastery updates."""
        
        print(f"\n{'='*50}")
        print("QUIZ COMPLETED")
        print(f"{'='*50}\n")
        
        # Score
        correct = sum(1 for r in session.responses if r.is_correct)
        total = len(session.questions)
        print(f"Score: {correct}/{total} ({session.score_percentage:.1f}%)\n")
        
        # Mastery updates
        print("Mastery Updates:")
        print("─" * 50)
        for concept, change in session.mastery_updates.items():
            state = knowledge_states.get(concept)
            if state:
                old_mastery = state.p_knowledge - change
                new_mastery = state.p_knowledge
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"{concept:20} {arrow} {old_mastery:.2%} → {new_mastery:.2%} ({change:+.2%})")
        
        # Time taken
        if session.completed_at:
            duration = (session.completed_at - session.started_at).total_seconds()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"\nTime taken: {minutes}m {seconds}s")
        
        print()

# Usage
if __name__ == "__main__":
    app = SimpleQuizApp()
    
    completed_session = app.run_quiz(
        user_id="student_001",
        concepts=["Python Basics", "Functions"],
        num_questions=3
    )
```

## Common Tasks

### Task 1: Target Knowledge Gaps

```python
from src.core.evaluation.generators.quiz_generator import QuizGenerator

generator = QuizGenerator()

# Generate quiz for specific gaps
quiz = generator.generate_quiz_from_knowledge_gaps(
    unknown_concepts=["Decorators", "Context Managers"],
    known_concepts=["Functions", "Classes"],
    num_questions=5
)
```

### Task 2: Assess Specific Mastery Level

```python
# Test if student has truly mastered a concept
quiz = generator.generate_mastery_quiz(
    concept="Functions",
    mastery_level=0.75,  # Generate hard questions
    num_questions=5
)
```

### Task 3: Review Quiz Results

```python
from src.core.evaluation.repository.quiz_repository import QuizRepository

repo = QuizRepository(storage_path="./quiz_data")

# Load a completed session
session = repo.load_session("session_id")

# Get all user sessions
user_sessions = repo.get_all_user_sessions("user123")

# Check concept performance
perf = repo.get_concept_performance("Functions")
print(f"Functions accuracy: {perf['accuracy']:.1%}")
print(f"Total attempts: {perf['total_attempts']}")
```

### Task 4: Export Quiz Data

```python
# Export all data
all_data = repo.export_all_data()

import json
with open("quiz_export.json", "w") as f:
    json.dump(all_data, f, indent=2, default=str)
```

## Troubleshooting

### Issue: "Quiz session not found"
```python
# Make sure session ID is correct
print(f"Session ID: {session.id}")

# Session should be added to engine when created
# If loading from storage, use:
session = repo.load_session(session_id)
```

### Issue: Question difficulty not adapting
```python
# Check mastery level is being updated
progress = engine.get_session_progress(session_id)
print(f"Mastery: {knowledge_states['Python Basics'].p_knowledge}")

# Difficulty is set when creating new questions or getting next question
next_q = engine.get_next_question(session_id, knowledge_states)
print(f"Question difficulty: {next_q.difficulty}")
```

### Issue: Feedback not appearing
```python
# Feedback is attached to response
response = engine.submit_response(...)
feedback = response.feedback

if feedback is None:
    print("Feedback generation failed")
else:
    print(feedback.explanation)
```

## Next Steps

1. **Read the full documentation**: See `QUIZ_ENGINE.md` for detailed features
2. **API Reference**: Check `QUIZ_ENGINE_API.md` for all methods
3. **Integrate with your app**: Follow the integration patterns in your application
4. **Customize questions**: Modify question templates in quiz generators
5. **Track analytics**: Use the repository to store and analyze quiz data
