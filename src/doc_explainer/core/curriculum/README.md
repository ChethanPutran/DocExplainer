"""
Curriculum Generator Module Documentation

The Curriculum Generator is the core orchestrator for creating personalized,
adaptive learning paths in Doc Explainer. It analyzes concept dependencies,
applies multiple sequencing strategies, and tracks user progress.

## Overview

The curriculum generator system consists of:

1. **ConceptDependencyResolver**: Analyzes concept graphs to extract prerequisites
   and dependencies, detecting circular dependencies and calculating depth metrics.

2. **CurriculumSequencer**: Base class for different sequencing strategies that
   order concepts according to specific pedagogical principles.

3. **CurriculumGenerator**: Main orchestrator that coordinates all components
   and provides a unified interface for generating and managing learning paths.

## Key Features

### Prerequisite-Ordered Learning Paths
- Analyzes concept dependency graphs using NetworkX
- Orders concepts respecting prerequisites
- Detects and handles circular dependencies gracefully
- Supports partial learning paths (starting from intermediate points)

### Adaptive Sequencing Based on User Progress
- Adjusts sequence based on current mastery levels
- Skips already-mastered concepts
- Inserts review/reinforcement at strategic points
- Predicts estimated time-to-completion with confidence intervals

### Multiple Curriculum Strategies

#### BREADTH_FIRST
Covers broad overview first, then deepens understanding.
- Good for learners who want the big picture
- Prioritizes concepts with fewer dependencies
- Allows quick exploration of the domain

#### DEPTH_FIRST
Deep dive into one area before moving to another.
- Good for learners who prefer mastery before breadth
- Follows dependency chains deeply
- Completes prerequisite chains before alternatives

#### ADAPTIVE
Dynamic reordering based on learning performance.
- Prioritizes concepts with lower mastery levels
- Respects dependencies
- Best suited for most learners

#### SPACED_REPETITION
Strategically timed reviews to optimize long-term retention.
- Interleaves concepts with strategic review points
- Based on learning science principles
- Optimizes for retention over time

#### MASTERY_BASED
Focus on weak areas and reinforce strong areas.
- Prioritizes concepts with low mastery
- Groups concepts by mastery level
- Ensures comprehensive coverage of weak areas

## Data Models

### CurriculumNode
Represents a single concept in the curriculum with metadata.

Fields:
- `concept_id`: Unique identifier
- `concept_name`: Human-readable name
- `dependencies`: Direct prerequisites
- `transitive_dependencies`: All prerequisites (direct and indirect)
- `estimated_time_minutes`: Estimated learning time
- `priority`: Priority score (0-1)
- `difficulty`: Difficulty score (0-1)
- `mastery_level`: User's current mastery (0-1)
- `dependency_depth`: Number of prerequisite levels

### LearningPath
Represents a complete personalized learning path.

Fields:
- `path_id`: Unique identifier
- `user_id`: User this path is for
- `concepts`: Ordered list of CurriculumNode objects
- `strategy`: The strategy used to generate this path
- `status`: Current progress state
- `progress`: Overall progress (0-1)
- `estimated_total_time_minutes`: Total estimated time
- `actual_time_spent_minutes`: Actual time spent so far
- `completed_concepts`: Set of completed concept IDs

### CurriculumStrategy
Enum specifying the sequencing strategy:
- BREADTH_FIRST
- DEPTH_FIRST
- ADAPTIVE
- SPACED_REPETITION
- MASTERY_BASED

### PathProgressState
Enum for learning path status:
- NOT_STARTED
- IN_PROGRESS
- COMPLETED
- PAUSED
- ABANDONED

## Usage Examples

### Basic Usage

```python
from src.core.orchestrator import CurriculumGenerator, CurriculumStrategy
from src.core.knowledge.models.graph import ConceptGraph
from src.core.user.services.user_profile_service import UserProfileService

# Initialize generator
concept_graph = ConceptGraph()  # Build your concept graph
user_service = UserProfileService(user_id="user123")

generator = CurriculumGenerator(
    concept_graph=concept_graph,
    user_profile_service=user_service
)

# Generate a learning path
path = generator.generate_learning_path(
    user_id="user123",
    strategy=CurriculumStrategy.ADAPTIVE
)

print(f"Generated path with {len(path.concepts)} concepts")
print(f"Estimated time: {path.estimated_total_time_minutes} minutes")
```

### Tracking Progress

```python
# Update progress when user completes a concept
generator.update_learning_path_progress(
    path=path,
    concept_id="python-basics",
    mastery_level=0.85,
    time_spent=45.0  # minutes
)

print(f"Progress: {path.progress * 100:.1f}%")
print(f"Time spent: {path.actual_time_spent_minutes:.0f} minutes")
```

### Estimating Time-to-Completion

```python
remaining_time, confidence = generator.estimate_time_to_completion(path)
print(f"Estimated remaining time: {remaining_time:.0f} minutes (confidence: {confidence:.1%})")
```

### Generating Partial Paths

```python
# Start from an intermediate concept, not from scratch
path = generator.generate_learning_path(
    user_id="user123",
    start_concepts=["python-basics"],  # Skip prerequisites of this
    strategy=CurriculumStrategy.DEPTH_FIRST
)
```

### Getting Strategy Recommendations

```python
recommended_strategy = generator.suggest_strategy(user_id="user123")
print(f"Recommended strategy: {recommended_strategy.value}")
```

## Integration with Existing Services

### UserProfileService
Used to:
- Get user's known/unknown concepts
- Retrieve mastery levels for personalization
- Access learning preferences
- Update concept mastery tracking

### ConceptGraph
Used to:
- Access concept relationships
- Analyze dependency structures
- Retrieve concept metadata
- Enable graph traversals

### MasteryEstimator (Optional)
Can be integrated to:
- Provide confidence-based mastery estimates
- Generate confidence intervals
- Track mastery progression over time

## Circular Dependency Handling

The system gracefully handles circular dependencies:

1. Detection: Uses topological sorting and cycle detection
2. Handling: Falls back to alternative ordering when cycles detected
3. Logging: Warns users about detected circular dependencies
4. Robustness: Still generates paths despite cycles (cycles excluded from ordering)

Example:
```python
try:
    path = generator.generate_learning_path(user_id="user1")
except CircularDependencyError as e:
    print(f"Circular dependency detected: {e}")
    # Try with a different strategy
    path = generator.generate_learning_path(
        user_id="user1",
        strategy=CurriculumStrategy.BREADTH_FIRST
    )
```

## Serialization and Persistence

All models support JSON serialization:

```python
# Serialize a learning path
path_data = path.to_dict()
json_string = json.dumps(path_data)

# Deserialize a learning path
loaded_path = LearningPath.from_dict(json.loads(json_string))
```

This enables:
- Saving paths to database
- Resuming paths across sessions
- Sharing paths with other users
- Backup and recovery

## Performance Considerations

### Caching
- Dependency resolver caches transitive dependencies
- Depth calculations are cached
- Cycle detection results are cached

### Scalability
- Handles graphs with hundreds of concepts
- Topological sorting is O(V + E) complexity
- Suitable for real-time path generation

### Memory
- Lazy evaluation of dependencies
- Efficient graph representations using NetworkX
- Minimal state storage

## Testing

The module includes comprehensive tests:

1. **Unit Tests** (`test_curriculum_generator.py`):
   - Dependency resolution algorithms
   - Individual sequencer strategies
   - Model serialization/deserialization
   - Edge cases and error handling

2. **Integration Tests** (`test_curriculum_integration.py`):
   - Full path generation workflow
   - Strategy recommendations
   - Progress tracking

Run tests with:
```bash
pytest src/core/orchestrator/tests/test_curriculum_generator.py -v
pytest src/core/orchestrator/tests/test_curriculum_integration.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            CurriculumGenerator (Orchestrator)               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ConceptDependencyResolver                            │  │
│  │  - Analyze concept graph                             │  │
│  │  - Resolve dependencies                              │  │
│  │  - Detect cycles                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Curriculum Sequencers (Strategy Pattern)            │  │
│  │  - BreadthFirstSequencer                             │  │
│  │  - DepthFirstSequencer                               │  │
│  │  - AdaptiveSequencer                                 │  │
│  │  - SpacedRepetitionSequencer                         │  │
│  │  - MasteryBasedSequencer                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Learning Path Generation & Management                │  │
│  │  - Generate paths                                    │  │
│  │  - Track progress                                    │  │
│  │  - Estimate completion                               │  │
│  │  - Suggest strategies                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      ConceptGraph  UserProfile   MasteryEstimator
      (Dependencies) (Preferences) (Levels)
```

## Future Enhancements

1. **Predictive Learning Time**: ML-based estimation of learning time per concept
2. **Intervention Recommendations**: Suggest additional resources or strategies
3. **Collaborative Filtering**: Recommend paths based on similar learners
4. **Competency Tracking**: Track domain-wide competency progression
5. **Prerequisite Validation**: Automated validation of prerequisite relationships
6. **Performance Analytics**: Detailed analytics on path effectiveness
7. **Export Formats**: Support for SCORM, xAPI, and other standards

## Troubleshooting

### "Circular dependency detected" Error
- Review concept relationships in your graph
- Break circular dependencies by removing or redefining relationships
- Consider using BREADTH_FIRST strategy as workaround

### Empty Learning Path
- Verify concepts exist in the graph
- Check that no exclude_concepts filter is too broad
- Ensure user profile service is properly initialized

### Inaccurate Time Estimates
- Collect more user data for better speed estimation
- Adjust estimated_time_minutes values in concepts
- Use custom mastery estimator for better accuracy

## References

- NetworkX Documentation: https://networkx.org/
- Topological Sorting: https://en.wikipedia.org/wiki/Topological_sorting
- Spaced Repetition: https://en.wikipedia.org/wiki/Spaced_repetition
- Learning Science Principles: https://www.apa.org/science/about/psa/learning

## License

Part of Doc Explainer project.
"""
