"""
Example Usage of Knowledge Mastery Estimation Module

This file demonstrates practical usage of the mastery_estimator module
for various scenarios.
"""

# Example 1: Basic BKT Usage
# ===========================

def example_bkt_basic():
    """Track a student's learning using Bayesian Knowledge Tracing."""
    from src.core.evaluation.mastery_estimator import BayesianKnowledgeTracer
    
    # Create tracer with custom parameters
    bkt = BayesianKnowledgeTracer(
        p_learn=0.35,   # 35% chance to learn from attempt
        p_guess=0.2,    # 20% chance of lucky guess
        p_slip=0.08     # 8% chance of careless error
    )
    
    # Student's response sequence
    initial_knowledge = 0.2
    responses = [True, True, False, True, True, True]
    
    # Trace through responses
    trajectory, final_knowledge = bkt.trace_sequence(initial_knowledge, responses)
    
    print(f"Initial: {trajectory[0]:.2f}")
    print(f"After 6 attempts: {final_knowledge:.2f}")
    print(f"Trajectory: {[f'{p:.2f}' for p in trajectory]}")


# Example 2: IRT-Based Assessment
# ================================

def example_irt_assessment():
    """Use IRT to estimate ability from assessment responses."""
    from src.core.evaluation.mastery_estimator import IRTEstimator
    
    # Create IRT estimator
    irt = IRTEstimator()
    
    # Configure item characteristics
    irt.set_concept_parameters(
        'quadratic_equations',
        difficulty=0.3,      # Moderately easy
        discrimination=1.2,  # Good discriminator
        guessing=0.25        # 4 options multiple choice
    )
    
    # Student's response pattern
    responses = [True, True, False, True, True]  # 4/5 correct
    
    # Estimate ability
    ability = irt.estimate_ability_mle(responses)
    mastery_probability = (ability + 3) / 6  # Convert to 0-1 scale
    
    print(f"Estimated Ability: {ability:.2f}")
    print(f"Mastery Probability: {mastery_probability:.2f}")


# Example 3: Complete Mastery Tracking
# =====================================

def example_complete_tracking():
    """Full workflow with mastery estimator."""
    from src.core.evaluation.mastery_estimator import (
        MasteryEstimator,
        InteractionResponse
    )
    from src.core.user.models.knowledge_state import KnowledgeState
    from src.core.knowledge.models.concept import Concept
    from datetime import datetime, timedelta
    
    # Initialize
    estimator = MasteryEstimator(
        mastery_threshold=0.9,
        novice_threshold=0.3,
        intermediate_threshold=0.7
    )
    
    concept = Concept(name='fractions')
    knowledge_state = KnowledgeState(concept=concept)
    
    # Simulate learning over time
    responses_over_time = [
        (False, "Initial attempt - struggles"),
        (False, "Second attempt - still incorrect"),
        (True, "Third attempt - improvement"),
        (True, "Fourth attempt - consistent"),
        (True, "Fifth attempt - mastering"),
        (True, "Sixth attempt - confident"),
    ]
    
    print(f"\nTracking mastery for concept: {concept.name}")
    print("-" * 50)
    
    for i, (is_correct, comment) in enumerate(responses_over_time, 1):
        # Create response
        response = InteractionResponse(
            is_correct=is_correct,
            timestamp=datetime.now() + timedelta(hours=i*2),
            confidence=0.8 if is_correct else 0.4
        )
        
        # Update mastery estimate
        knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        # Get current estimate
        estimate = estimator.get_mastery_estimate('fractions', knowledge_state)
        
        print(f"\nAttempt {i}: {comment}")
        print(f"  Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        print(f"  Knowledge: {estimate['bkt_mastery']:.2%}")
        print(f"  Level: {estimate['mastery_level']}")
        print(f"  Accuracy: {estimate['accuracy']:.2%}")
        
        if estimate['is_mastered']:
            print(f"  🎉 MASTERED!")
            break


# Example 4: Confidence Intervals
# ================================

def example_confidence_intervals():
    """Calculate confidence intervals for mastery estimates."""
    from src.core.evaluation.mastery_estimator import ConfidenceEstimator
    
    # Scenario: Student got 75/100 on a test
    estimator = ConfidenceEstimator()
    
    # Wilson score CI (better for binary outcomes)
    ci = estimator.wilson_ci(successes=75, trials=100, ci=0.95)
    
    print(f"Wilson 95% Confidence Interval:")
    print(f"  Point Estimate: {ci.point_estimate:.2%}")
    print(f"  95% CI: [{ci.lower:.2%}, {ci.upper:.2%}]")
    print(f"  Interpretation: We are 95% confident the true mastery")
    print(f"                  is between {ci.lower:.2%} and {ci.upper:.2%}")


# Example 5: Learning Timeline Analysis
# =======================================

def example_timeline_analysis():
    """Analyze learning progression and predict time to mastery."""
    from src.core.evaluation.mastery_estimator import (
        MasteryEstimator,
        InteractionResponse
    )
    from src.core.user.models.knowledge_state import KnowledgeState
    from src.core.knowledge.models.concept import Concept
    from datetime import datetime, timedelta
    
    estimator = MasteryEstimator()
    concept = Concept(name='probability')
    knowledge_state = KnowledgeState(concept=concept)
    
    # Simulate learning over several days
    num_sessions = 10
    responses_per_session = 5
    success_rate = 0.4  # Starts at 40%
    
    print(f"\nTracking learning progress over {num_sessions} sessions")
    print("=" * 50)
    
    for session in range(num_sessions):
        # Simulate improving success rate
        current_success_rate = min(0.95, success_rate + (session * 0.06))
        
        for _ in range(responses_per_session):
            is_correct = __import__('random').random() < current_success_rate
            response = InteractionResponse(
                is_correct=is_correct,
                timestamp=datetime.now() - timedelta(hours=(num_sessions - session) * 2)
            )
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        if session % 3 == 0:  # Print every 3 sessions
            timeline = estimator.get_mastery_timeline('probability')
            if timeline:
                print(f"\nAfter session {session + 1}:")
                print(f"  Current Mastery: {timeline['current_mastery']:.2%}")
                if timeline['learning_velocity']:
                    print(f"  Learning Velocity: {timeline['learning_velocity']:.4f} per hour")
                if timeline['time_to_mastery']:
                    hours = timeline['time_to_mastery'].total_seconds() / 3600
                    print(f"  Estimated time to mastery: {hours:.1f} hours")


# Example 6: Multi-Concept Learning
# ==================================

def example_multi_concept():
    """Track learning across multiple concepts."""
    from src.core.evaluation.mastery_estimator import (
        MasteryEstimator,
        InteractionResponse
    )
    from src.core.user.models.knowledge_state import KnowledgeState
    from src.core.knowledge.models.concept import Concept
    
    estimator = MasteryEstimator()
    
    concepts = ['addition', 'subtraction', 'multiplication', 'division']
    knowledge_states = {
        name: KnowledgeState(concept=Concept(name=name))
        for name in concepts
    }
    
    # Simulate different learning patterns
    patterns = {
        'addition': [True] * 8,              # Fast learner
        'subtraction': [True, False] * 4,   # Inconsistent
        'multiplication': [False] * 3 + [True] * 5,  # Struggles initially
        'division': [False] * 8,            # Needs more help
    }
    
    print("\nMulti-Concept Learning Summary")
    print("=" * 50)
    
    for concept, responses in patterns.items():
        for is_correct in responses:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_states[concept] = estimator.update_from_response(
                knowledge_states[concept], response
            )
    
    # Summary
    print(f"\n{'Concept':<15} {'Mastery':<10} {'Level':<15} {'Status':<10}")
    print("-" * 50)
    
    for concept in concepts:
        estimate = estimator.get_mastery_estimate(concept, knowledge_states[concept])
        status = "✓ Mastered" if estimate['is_mastered'] else "⏳ Learning"
        print(f"{concept:<15} {estimate['bkt_mastery']:<10.2%} "
              f"{estimate['mastery_level']:<15} {status:<10}")


# Example 7: BKT vs IRT Comparison
# =================================

def example_bkt_vs_irt():
    """Compare mastery estimates from BKT and IRT."""
    from src.core.evaluation.mastery_estimator import (
        MasteryEstimator,
        InteractionResponse
    )
    from src.core.user.models.knowledge_state import KnowledgeState
    from src.core.knowledge.models.concept import Concept
    
    estimator = MasteryEstimator()
    concept = Concept(name='calculus')
    knowledge_state = KnowledgeState(concept=concept)
    
    # Configure IRT parameters
    estimator.set_irt_parameters(
        'calculus',
        difficulty=0.2,
        discrimination=1.3,
        guessing=0.1
    )
    
    # Generate responses
    responses = [True, True, False, True, True, True, True, False, True, True]
    
    for is_correct in responses:
        response = InteractionResponse(is_correct=is_correct)
        knowledge_state = estimator.update_from_response(knowledge_state, response)
    
    # Compare estimates
    comparison = estimator.compare_bkt_vs_irt('calculus', knowledge_state)
    
    print("\nBKT vs IRT Comparison")
    print("=" * 50)
    print(f"BKT Estimate:     {comparison['bkt']:.2%}")
    print(f"IRT Estimate:     {comparison['irt']:.2%}")
    print(f"Difference:       {comparison['difference']:.2%}")
    print(f"Agreement Score:  {comparison['agreement']:.2%}")
    print(f"Average:          {comparison['estimate']:.2%}")
    
    if comparison['agreement'] > 0.9:
        print("\n✓ Excellent agreement between models")
    elif comparison['agreement'] > 0.7:
        print("\n≈ Good agreement between models")
    else:
        print("\n⚠ Divergent estimates - investigate further")


if __name__ == "__main__":
    print("=" * 70)
    print("KNOWLEDGE MASTERY ESTIMATION - USAGE EXAMPLES")
    print("=" * 70)
    
    print("\n\n1. BASIC BKT USAGE")
    print("-" * 70)
    try:
        example_bkt_basic()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n\n2. IRT-BASED ASSESSMENT")
    print("-" * 70)
    try:
        example_irt_assessment()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n\n3. CONFIDENCE INTERVALS")
    print("-" * 70)
    try:
        example_confidence_intervals()
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "=" * 70)
    print("Examples demonstrating:")
    print("  • Bayesian Knowledge Tracing (BKT)")
    print("  • Item Response Theory (IRT)")
    print("  • Confidence intervals")
    print("  • Timeline tracking")
    print("  • Multi-concept learning")
    print("  • Model comparison")
    print("=" * 70)
