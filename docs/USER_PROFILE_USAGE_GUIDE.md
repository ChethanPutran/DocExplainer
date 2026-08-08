# User Profile Service - Usage Guide

## Quick Start

### Basic Setup

```python
from src.core.user import UserProfileService, UserProfileRepository

# Create a service for a user
service = UserProfileService("user_123")

# Create a repository for persistence
repo = UserProfileRepository("data/profiles/")
```

## 1. Concept Classification

### Classify Concepts as Known/Unknown

```python
# Classify a concept as known or unknown
is_python_known = service.classify_concept("Python", 0.85)  # True (>= 0.7)
is_rust_known = service.classify_concept("Rust", 0.25)       # False (< 0.7)

# Quick lookup
if service.is_concept_known("Python"):
    print("User knows Python")

# Get all known concepts
known = service.get_known_concepts()  # Dict[str, ConceptMastery]

# Get all unknown concepts  
unknown = service.get_unknown_concepts()  # Dict[str, ConceptMastery]

# Get set of known concept names for membership testing
known_set = service.get_known_concepts_set()  # Set[str]
if "Python" in known_set:
    print("Fast membership test")
```

### Custom Thresholds

```python
# Change the threshold for what counts as "known"
service.set_known_threshold(0.5)  # More permissive
assert service.is_concept_known("Rust")  # 0.25 >= 0.5? No, but example shows intent

# Use custom threshold for single classification
is_known = service.classify_concept("Go", 0.6, threshold=0.5)  # Uses 0.5, not 0.7
```

## 2. Mastery Level Management

### Calculate Mastery Levels

```python
from src.core.user.models.user_profile import MasteryLevel

# Mastery levels:
# Novice:      p_knowledge < 0.3
# Intermediate: 0.3 <= p_knowledge < 0.7
# Expert:      0.7 <= p_knowledge < 0.9
# Mastered:    p_knowledge >= 0.9

level = service.calculate_mastery_level(0.8)  # Returns MasteryLevel.EXPERT
assert level == MasteryLevel.EXPERT
```

### Update and Track Mastery

```python
# Update a concept's mastery
service.update_concept_mastery("Python", p_knowledge=0.95, confidence=0.9)

# Get concepts at specific mastery level
mastered = service.get_concepts_by_mastery(MasteryLevel.MASTERED)
expert = service.get_concepts_by_mastery(MasteryLevel.EXPERT)

# Get distribution across all levels
distribution = service.get_mastery_distribution()
# {
#     'novice': 2,
#     'intermediate': 5,
#     'expert': 3,
#     'mastered': 1
# }

# Get average mastery
avg = service.get_average_mastery()  # 0-1 score

# Get concepts for learning (novice + intermediate)
to_learn = service.get_concepts_for_learning()

# Get advanced concepts
advanced = service.get_advanced_concepts(include_expert=True)
```

## 3. Learning Preferences

### Get Current Preferences

```python
prefs = service.get_preferences()
print(f"Learning pace: {prefs.learning_pace}")
print(f"Explanation depth: {prefs.explanation_depth}")
```

### Update Preferences

```python
# Update multiple at once
service.update_preferences(
    explanation_depth=4,           # 1-5
    learning_pace='fast',          # 'slow', 'normal', 'fast'
    preferred_modality='visual',   # 'text', 'visual', 'interactive'
    quiz_frequency='often',        # 'never', 'rarely', 'sometimes', 'often', 'always'
    language_preference='es',      # ISO 639-1 code
    show_hints=False
)

# Or set individually
service.set_explanation_depth(5)     # ExplanationDepth.COMPREHENSIVE
service.set_learning_pace('fast')    # LearningPace.FAST
service.set_preferred_modality('visual')  # PreferredModality.VISUAL
service.set_quiz_frequency('often')  # QuizFrequency.OFTEN
service.set_language_preference('fr')
```

## 4. Profile Statistics

### Get Learning Statistics

```python
stats = service.get_learning_statistics()
# {
#     'total_concepts': 10,
#     'known_concepts': 6,
#     'unknown_concepts': 4,
#     'average_mastery': 0.65,
#     'mastery_distribution': {...},
#     'novice_count': 2,
#     'intermediate_count': 3,
#     'expert_count': 2,
#     'mastered_count': 1
# }

# Get complete profile summary
summary = service.get_profile_summary()
# {
#     'user_id': 'user_123',
#     'total_concepts': 10,
#     'known_concepts_count': 6,
#     'average_mastery': 0.65,
#     'mastery_distribution': {...},
#     'preferences': {...},
#     'created_at': '2024-01-15T10:30:00',
#     'updated_at': '2024-01-15T14:45:00',
#     'last_active': '2024-01-15T14:45:00'
# }
```

## 5. Profile Persistence

### Save and Retrieve

```python
# Save profile to disk
repo.save_profile(service.profile)

# Load profile from disk
loaded_profile = repo.get_profile("user_123")

# Check if profile exists
if repo.profile_exists("user_123"):
    print("Profile found")

# Delete profile
repo.delete_profile("user_123")

# List all users
user_ids = repo.list_user_ids()
```

### Export and Import

```python
# Export profile to file
repo.export_profile("user_123", "exports/user_123.json")

# Import profile from file
imported_profile = repo.import_profile("exports/user_123.json")

# Automatic backups are created on update
# Manage backups
repo.cleanup_old_backups(max_backups_per_user=5)
```

## 6. Advanced Queries

### Find Users by Statistics

```python
# Find users with mastery in range
advanced_users = repo.find_profiles_by_mastery_range(
    min_mastery=0.7,
    max_mastery=1.0
)

# Find users by preference
fast_learners = repo.get_profiles_by_preference(
    'learning_pace',
    'fast'
)

visual_learners = repo.get_profiles_by_preference(
    'preferred_modality',
    'visual'
)
```

### Get System Statistics

```python
# Stats for single user
user_stats = repo.get_profile_statistics("user_123")

# Stats for all users
all_stats = repo.get_all_statistics()
# {
#     'total_users': 150,
#     'profiles': [
#         {'user_id': 'user_123', ...},
#         ...
#     ]
# }
```

## 7. Complete Example Workflow

```python
from src.core.user import UserProfileService, UserProfileRepository

# Initialize
service = UserProfileService("user_456")
repo = UserProfileRepository()

# Record learning interactions
interactions = [
    ("Python", 0.85),
    ("JavaScript", 0.72),
    ("Java", 0.65),
    ("Rust", 0.15),
    ("Go", 0.25),
]

for concept, confidence in interactions:
    service.classify_concept(concept, confidence)
    service.update_concept_mastery(concept, confidence)

# Configure preferences
service.update_preferences(
    explanation_depth=4,
    learning_pace='normal',
    preferred_modality='text',
    quiz_frequency='sometimes',
    language_preference='en'
)

# Get recommendations
to_learn = service.get_concepts_for_learning()
print(f"Concepts to learn: {list(to_learn.keys())}")  # Java, Rust, Go

advanced = service.get_advanced_concepts()
print(f"Advanced concepts: {list(advanced.keys())}")  # Python, JavaScript

# Get statistics
stats = service.get_learning_statistics()
print(f"Average mastery: {stats['average_mastery']:.2f}")

# Save profile
repo.save_profile(service.profile)
print("Profile saved!")

# Later: retrieve and continue
retrieved = repo.get_profile("user_456")
service2 = UserProfileService("user_456")
service2.set_profile(retrieved)

# Verify data persisted
assert "Python" in service2.profile.known_concepts
print("Profile restored successfully!")
```

## Data Model Reference

### ConceptMastery

```python
# Fields
concept_name: str
p_knowledge: float  # 0-1, probability of knowing
mastery_level: MasteryLevel  # Auto-calculated
first_seen: datetime
last_seen: datetime
times_seen: int
confidence: float  # 0-1, confidence in estimate

# Methods
update_knowledge(p_knowledge)  # Update and recalculate level
is_known(threshold=0.7) -> bool  # Check if above threshold
```

### UserProfile

```python
# Fields
user_id: str
known_concepts: Dict[str, ConceptMastery]
unknown_concepts: Set[str]
preferences: LearningPreferences
created_at: datetime
updated_at: datetime
last_active: datetime

# Methods
get_known_concepts(threshold=0.7)
get_unknown_concepts(threshold=0.7)
get_concepts_by_mastery(level: MasteryLevel)
update_concept_mastery(name, p_knowledge, confidence)
get_mastery_distribution()
get_profile_summary()
```

### LearningPreferences

```python
# Fields (all optional, have defaults)
explanation_depth: ExplanationDepth  # 1-5 scale
learning_pace: LearningPace  # slow, normal, fast
preferred_modality: PreferredModality  # text, visual, interactive
quiz_frequency: QuizFrequency  # never, rarely, sometimes, often, always
language_preference: str  # ISO 639-1 code
auto_advanced_quizzes: bool
detailed_feedback: bool
show_hints: bool
```

## Error Handling

```python
# Invalid thresholds
try:
    service.set_known_threshold(1.5)  # Raises ValueError
except ValueError as e:
    print(f"Invalid threshold: {e}")

# Non-existent profiles
profile = repo.get_profile("nonexistent")
if profile is None:
    print("Profile not found")

# File I/O errors
try:
    repo.import_profile("/bad/path/profile.json")
except Exception as e:
    print(f"Import failed: {e}")
```

## Performance Tips

1. **Use quick lookups for membership tests**
   ```python
   # Fast: O(1)
   known_set = service.get_known_concepts_set()
   if "Python" in known_set:
       ...
   
   # Slower: O(n)
   known = service.get_known_concepts()
   if "Python" in known:
       ...
   ```

2. **Batch updates when possible**
   ```python
   # Good
   for concept, score in scores.items():
       service.update_concept_mastery(concept, score)
   repo.save_profile(service.profile)
   
   # Not as efficient
   for concept, score in scores.items():
       service.update_concept_mastery(concept, score)
       repo.save_profile(service.profile)  # Saves each time!
   ```

3. **Cache is automatically managed**
   - Cache invalidated on threshold changes
   - Profile cache cleared on demand
   - Use `repo.reload_profile()` to bypass cache

## Integration with Existing Code

The implementation integrates seamlessly with existing Doc Explainer code:

```python
from src.core.user import User, UserProfileService, UserManager

# Existing User object
user = User(user_id="user_789")

# New profile service
profile_service = UserProfileService("user_789")

# Can work together
profile_service.classify_concept("Python", user.get_confidence("Python"))

# Update user's knowledge based on profile
for concept, mastery in profile_service.profile.known_concepts.items():
    # Update user's knowledge state if needed
    pass
```

---

For more details, see IMPLEMENTATION_SUMMARY.txt
