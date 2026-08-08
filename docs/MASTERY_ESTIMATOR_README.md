"""
Knowledge Mastery Estimation Module

Documentation for src/core/evaluation/mastery_estimator.py

## Overview

The mastery_estimator module provides a comprehensive framework for estimating and tracking
knowledge mastery using advanced psychometric models and statistical techniques.

## Key Components

### 1. Bayesian Knowledge Tracing (BKT)

Implements the classic BKT algorithm for tracking knowledge probability.

**Parameters:**
- `p_knowledge`: Probability student knows the concept (0-1)
- `p_learn`: Probability of learning from an attempt (0-1)
- `p_guess`: Probability of correct guess without knowledge (0-1)
- `p_slip`: Probability of slipping - error despite knowing (0-1)

**Model:**
```
P(correct) = p_knowledge × (1 - p_slip) + (1 - p_knowledge) × p_guess

Update on correct:
  P(L_t) = P(L_{t-1}) + (1 - P(L_{t-1})) × p_learn

Update on incorrect:
  P(L_t) = P(L_{t-1}) × (1 - p_slip)
```

**Example:**
```python
from src.core.evaluation.mastery_estimator import BayesianKnowledgeTracer

bkt = BayesianKnowledgeTracer(p_learn=0.3, p_guess=0.2, p_slip=0.1)

# Update knowledge based on responses
knowledge = 0.3
responses = [True, True, False, True]

trajectory, final_knowledge = bkt.trace_sequence(knowledge, responses)
print(f"Final knowledge: {final_knowledge:.2f}")
```

### 2. Item Response Theory (IRT) - 3-Parameter Logistic Model

Implements the 3-parameter logistic model for adaptive assessment.

**Parameters:**
- `θ`: Student ability
- `α`: Item discrimination (steepness of curve)
- `β`: Item difficulty
- `γ`: Guessing parameter (lower asymptote)

**Model:**
```
P(correct | θ) = γ + (1 - γ) × Λ(α × (θ - β))

where Λ(x) = 1 / (1 + e^(-x))  (logistic function)
```

**Example:**
```python
from src.core.evaluation.mastery_estimator import IRTEstimator

irt = IRTEstimator()

# Set parameters for a concept
irt.set_concept_parameters('algebra', difficulty=0.5, 
                          discrimination=1.2, guessing=0.15)

# Estimate ability from response pattern
responses = [True, True, False, True, True]
ability = irt.estimate_ability_mle(responses)
mastery = (ability + 3) / 6  # Convert to 0-1 scale
```

### 3. Confidence Intervals & Uncertainty Quantification

Provides two methods for calculating confidence intervals:

#### Bootstrap Method
- Resamples data to estimate sampling distribution
- Non-parametric, works for any distribution
- More accurate for small samples

#### Wilson Score Method
- Specifically designed for binary outcomes
- More accurate than normal approximation
- Better for extreme proportions

**Example:**
```python
from src.core.evaluation.mastery_estimator import ConfidenceEstimator

# Bootstrap CI
data = [0.7, 0.75, 0.8, 0.78, 0.82]
ci = ConfidenceEstimator.bootstrap_ci(data, ci=0.95)
print(f"CI: [{ci.lower:.3f}, {ci.point_estimate:.3f}, {ci.upper:.3f}]")

# Wilson CI for binary outcomes
ci = ConfidenceEstimator.wilson_ci(successes=8, trials=10, ci=0.95)
```

### 4. Mastery Timeline Tracking

Tracks mastery progression over time with automated learning pattern detection.

**Features:**
- Stores snapshots of mastery state
- Detects learning spikes and plateaus
- Calculates learning velocity
- Estimates time-to-mastery

**Example:**
```python
from src.core.evaluation.mastery_estimator import MasteryTimeline, MasterySnapshot
from datetime import datetime

timeline = MasteryTimeline()

# Add snapshots as user learns
snapshot = MasterySnapshot(
    timestamp=datetime.now(),
    p_knowledge=0.65,
    n_attempts=20,
    n_correct=13,
    accuracy=0.65
)
timeline.add_snapshot(snapshot)

# Get learning analytics
velocity = timeline.get_learning_velocity(window_hours=24)
time_to_mastery = timeline.estimate_time_to_mastery(target=0.9)
summary = timeline.get_summary()
```

### 5. Main MasteryEstimator

Unified interface combining BKT and IRT for comprehensive mastery estimation.

**Key Methods:**
- `update_from_response()`: Update knowledge state based on user response
- `get_mastery_estimate()`: Get comprehensive estimate with confidence
- `get_mastery_timeline()`: Get progression data
- `compare_bkt_vs_irt()`: Compare two estimation approaches
- `set_irt_parameters()`: Configure IRT for specific concepts

**Example:**
```python
from src.core.evaluation.mastery_estimator import (
    MasteryEstimator, 
    InteractionResponse
)
from src.core.user.models.knowledge_state import KnowledgeState
from src.core.knowledge.models.concept import Concept

# Initialize
estimator = MasteryEstimator(mastery_threshold=0.9)
concept = Concept(name='algebra')
knowledge_state = KnowledgeState(concept=concept)

# Process responses
response = InteractionResponse(is_correct=True)
knowledge_state = estimator.update_from_response(knowledge_state, response)

# Get comprehensive estimate
estimate = estimator.get_mastery_estimate('algebra', knowledge_state)
print(f"Mastery: {estimate['bkt_mastery']:.2f}")
print(f"Mastery Level: {estimate['mastery_level']}")
print(f"Is Mastered: {estimate['is_mastered']}")
print(f"Confidence: {estimate['confidence_interval']}")
```

## Mastery Classifications

The system classifies knowledge into three levels:

- **Novice** (< 0.3): Limited understanding
- **Intermediate** (0.3 - 0.7): Developing understanding
- **Expert** (> 0.7): Strong understanding
- **Mastered** (≥ 0.9): Meets mastery threshold

## Architecture

```
MasteryEstimator (main interface)
├── BayesianKnowledgeTracer (BKT algorithm)
├── IRTEstimator (3PL IRT model)
├── ConfidenceEstimator (uncertainty quantification)
├── MasteryTimeline (progression tracking)
└── ConfidenceInterval (uncertainty data structure)
```

## Integration with Existing System

The module:
- **Uses** KnowledgeState from `src/core/user/models/knowledge_state.py`
- **Uses** Concept from `src/core/knowledge/models/concept.py`
- **Does NOT modify** existing models
- **Is compatible** with existing evaluation modules

## Dependencies

- `numpy`: Numerical computations
- `scipy.stats`: Statistical functions
- Python 3.8+

## Testing

Comprehensive test suite in `src/core/evaluation/tests/test_mastery_estimator.py`

**Test Coverage:**
- BKT algorithm (observation probability, knowledge updates, sequence tracing)
- IRT model (logistic function, 3PL model, ability estimation)
- Confidence intervals (bootstrap, Wilson score)
- Mastery timeline (velocity, time-to-mastery, event detection)
- Main estimator (integration tests, multi-concept tracking)

**Run Tests:**
```bash
pytest src/core/evaluation/tests/test_mastery_estimator.py -v
```

## Performance Characteristics

- **BKT Update**: O(1) - Constant time
- **IRT Ability Estimation**: O(n × iterations) where n = number of responses
- **Bootstrap CI**: O(n × bootstrap_samples) 
- **Wilson CI**: O(1) - Closed-form calculation

## Configuration

All thresholds are configurable:

```python
estimator = MasteryEstimator(
    mastery_threshold=0.9,      # Threshold for mastery status
    novice_threshold=0.3,        # Novice level upper bound
    intermediate_threshold=0.7   # Intermediate level upper bound
)
```

## Parameters Tuning Guide

**BKT Parameters:**
- `p_learn`: Higher for learnable concepts, lower for difficult ones
  - Default: 0.3 (typical classroom setting)
  - Range: 0.1 - 0.5
  
- `p_guess`: Higher for multiple choice, lower for constructed response
  - Default: 0.2
  - Range: 0.05 - 0.5
  
- `p_slip`: Probability of careless mistakes
  - Default: 0.1
  - Range: 0.05 - 0.2

**IRT Parameters:**
- `discrimination (α)`: How well item differentiates ability levels
  - Typical: 0.8 - 2.0
  - Higher = steeper, more discriminating
  
- `difficulty (β)`: Location of steepest point (where P = 0.5 without guessing)
  - Typical: -2 to 2
  - Negative = easier, Positive = harder
  
- `guessing (γ)`: Lower asymptote for random guessing
  - Typical: 0.15 - 0.25 for MC with 4-5 options
  - 0 for constructed response

## Caveats and Limitations

1. **BKT Limitations:**
   - Assumes fixed parameters across students
   - Two-state model (knows/doesn't know) is simplification
   - Not ideal for continuous skill development

2. **IRT Limitations:**
   - Requires calibration data for accurate parameters
   - May overfit with limited responses
   - Assumes unidimensional ability

3. **Timeline Tracking:**
   - Velocity estimates require sufficient data points
   - Time-to-mastery extrapolation assumes constant learning rate
   - May not account for forgetting

## Future Enhancements

- Dynamic parameter adaptation based on learner profile
- Multi-dimensional IRT models
- Spaced repetition integration
- Forgetting curves (Ebbinghaus)
- Bayesian parameter learning
- Concept prerequisites/dependency modeling

## References

- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge.
- Lord, F. M. (1980). Applications of Item Response Theory to Practical Testing Problems.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion.
"""
