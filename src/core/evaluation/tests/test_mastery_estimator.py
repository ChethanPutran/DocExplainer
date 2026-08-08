"""
Comprehensive tests for Knowledge Mastery Estimation module.

Tests cover:
- Bayesian Knowledge Tracing (BKT) algorithm
- Item Response Theory (IRT) mastery estimation
- Confidence interval calculations
- Mastery timeline tracking
- Complete mastery estimator integration
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.core.evaluation.mastery_estimator import (
    BayesianKnowledgeTracer,
    IRTEstimator,
    ConfidenceEstimator,
    ConfidenceInterval,
    MasteryLevel,
    MasteryTimeline,
    MasterySnapshot,
    MasteryEstimator,
    InteractionResponse
)
from src.core.user.models.knowledge_state import KnowledgeState
from src.core.knowledge.models.concept import Concept


class TestBayesianKnowledgeTracer:
    """Test suite for BKT algorithm."""
    
    def test_initialization(self):
        """Test BKT initialization with default parameters."""
        bkt = BayesianKnowledgeTracer()
        assert bkt.p_learn == 0.3
        assert bkt.p_guess == 0.2
        assert bkt.p_slip == 0.1
    
    def test_initialization_custom_params(self):
        """Test BKT initialization with custom parameters."""
        bkt = BayesianKnowledgeTracer(p_learn=0.4, p_guess=0.15, p_slip=0.05)
        assert bkt.p_learn == 0.4
        assert bkt.p_guess == 0.15
        assert bkt.p_slip == 0.05
    
    def test_observation_probability_correct_response(self):
        """Test observation probability calculation for correct response."""
        bkt = BayesianKnowledgeTracer(p_guess=0.2, p_slip=0.1)
        
        # If student knows for sure (p_knowledge=1)
        prob = bkt.observation_probability(1.0, is_correct=True)
        assert abs(prob - 0.9) < 1e-6  # Should be 1 * (1 - 0.1) = 0.9
        
        # If student doesn't know (p_knowledge=0)
        prob = bkt.observation_probability(0.0, is_correct=True)
        assert abs(prob - 0.2) < 1e-6  # Should be 0.2 (pure guess)
    
    def test_observation_probability_incorrect_response(self):
        """Test observation probability for incorrect response."""
        bkt = BayesianKnowledgeTracer(p_guess=0.2, p_slip=0.1)
        
        # If student knows for sure (p_knowledge=1)
        prob = bkt.observation_probability(1.0, is_correct=False)
        assert abs(prob - 0.1) < 1e-6  # Should slip with p=0.1
        
        # If student doesn't know (p_knowledge=0)
        prob = bkt.observation_probability(0.0, is_correct=False)
        assert abs(prob - 0.8) < 1e-6  # Should guess incorrectly with p=0.8
    
    def test_update_knowledge_correct_response(self):
        """Test knowledge update on correct response."""
        bkt = BayesianKnowledgeTracer()
        
        # Starting with low knowledge
        initial = 0.1
        updated = bkt.update_knowledge(initial, is_correct=True)
        
        # Knowledge should increase
        assert updated > initial
        assert updated <= 1.0
    
    def test_update_knowledge_incorrect_response(self):
        """Test knowledge update on incorrect response."""
        bkt = BayesianKnowledgeTracer()
        
        # Starting with high knowledge
        initial = 0.8
        updated = bkt.update_knowledge(initial, is_correct=False)
        
        # Knowledge should decrease
        assert updated < initial
        assert updated >= 0.0
    
    def test_update_knowledge_bounds(self):
        """Test that knowledge stays within [0, 1] bounds."""
        bkt = BayesianKnowledgeTracer()
        
        # Test with extreme values
        for initial in [0.0, 0.5, 1.0]:
            for is_correct in [True, False]:
                updated = bkt.update_knowledge(initial, is_correct)
                assert 0.0 <= updated <= 1.0
    
    def test_trace_sequence(self):
        """Test tracing through a sequence of responses."""
        bkt = BayesianKnowledgeTracer()
        
        # All correct responses - should increase knowledge
        responses = [True, True, True, True, True]
        trajectory, final = bkt.trace_sequence(0.1, responses)
        
        assert len(trajectory) == 6  # Initial + 5 updates
        assert trajectory[0] == 0.1  # Initial
        assert final > trajectory[0]  # Should have increased
        assert final <= 1.0
    
    def test_trace_sequence_all_incorrect(self):
        """Test sequence with all incorrect responses."""
        bkt = BayesianKnowledgeTracer()
        
        # All incorrect - should decrease knowledge
        responses = [False, False, False, False, False]
        trajectory, final = bkt.trace_sequence(0.8, responses)
        
        assert len(trajectory) == 6
        assert final < trajectory[0]
        assert final >= 0.0
    
    def test_trace_sequence_mixed(self):
        """Test sequence with mixed correct/incorrect."""
        bkt = BayesianKnowledgeTracer()
        
        responses = [True, False, True, True, False]
        trajectory, final = bkt.trace_sequence(0.3, responses)
        
        assert len(trajectory) == 6
        assert all(0 <= t <= 1 for t in trajectory)


class TestIRTEstimator:
    """Test suite for IRT estimator."""
    
    def test_initialization(self):
        """Test IRT initialization."""
        irt = IRTEstimator()
        assert irt.concept_params == {}
    
    def test_logistic_function(self):
        """Test logistic function."""
        irt = IRTEstimator()
        
        # Test known values
        assert abs(irt.logistic(0) - 0.5) < 1e-6
        assert irt.logistic(100) > 0.999
        assert irt.logistic(-100) < 0.001
    
    def test_response_probability(self):
        """Test 3PL response probability calculation."""
        irt = IRTEstimator()
        
        # High ability, low difficulty, no guessing
        prob = irt.response_probability(2.0, 1.0, 0.0, 0.0)
        assert prob > 0.5
        
        # Low ability, high difficulty, no guessing
        prob = irt.response_probability(-2.0, 1.0, 2.0, 0.0)
        assert prob < 0.5
        
        # With guessing
        prob = irt.response_probability(-10.0, 1.0, 0.0, 0.2)
        assert abs(prob - 0.2) < 0.01  # Should be approximately equal to guessing param
    
    def test_estimate_ability_mle_single_response(self):
        """Test MLE ability estimation with single response."""
        irt = IRTEstimator()
        
        # Single correct response
        ability = irt.estimate_ability_mle([True])
        assert -3 <= ability <= 3
        
        # Single incorrect response
        ability = irt.estimate_ability_mle([False])
        assert -3 <= ability <= 3
    
    def test_estimate_ability_mle_all_correct(self):
        """Test MLE with all correct responses."""
        irt = IRTEstimator()
        
        responses = [True] * 10
        ability = irt.estimate_ability_mle(responses)
        
        # Should have positive ability
        assert ability > 0
        assert ability <= 3
    
    def test_estimate_ability_mle_all_incorrect(self):
        """Test MLE with all incorrect responses."""
        irt = IRTEstimator()
        
        responses = [False] * 10
        ability = irt.estimate_ability_mle(responses)
        
        # Should have negative ability
        assert ability < 0
        assert ability >= -3
    
    def test_estimate_ability_mle_mixed(self):
        """Test MLE with mixed responses."""
        irt = IRTEstimator()
        
        responses = [True, False, True, True, False]
        ability = irt.estimate_ability_mle(responses)
        
        # With more correct than incorrect, should be positive
        assert -3 <= ability <= 3
    
    def test_estimate_ability_empty_responses(self):
        """Test MLE with empty responses."""
        irt = IRTEstimator()
        
        ability = irt.estimate_ability_mle([])
        assert ability == 0.0
    
    def test_set_concept_parameters(self):
        """Test setting concept IRT parameters."""
        irt = IRTEstimator()
        
        irt.set_concept_parameters('algebra', difficulty=0.5,
                                   discrimination=1.2, guessing=0.15)
        
        params = irt.get_concept_parameters('algebra')
        assert params['difficulty'] == 0.5
        assert params['discrimination'] == 1.2
        assert params['guessing'] == 0.15
    
    def test_get_default_parameters(self):
        """Test getting default parameters for unknown concept."""
        irt = IRTEstimator()
        
        params = irt.get_concept_parameters('unknown_concept')
        assert params['difficulty'] == 0.0
        assert params['discrimination'] == 1.0
        assert params['guessing'] == 0.2


class TestConfidenceEstimator:
    """Test suite for confidence interval estimation."""
    
    def test_bootstrap_ci_basic(self):
        """Test bootstrap confidence interval."""
        data = [0.7, 0.8, 0.75, 0.78, 0.82]
        ci = ConfidenceEstimator.bootstrap_ci(data, ci=0.95)
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.confidence_level == 0.95
    
    def test_bootstrap_ci_empty_data(self):
        """Test bootstrap CI with empty data."""
        ci = ConfidenceEstimator.bootstrap_ci([])
        
        assert ci.lower == 0.0
        assert ci.point_estimate == 0.0
        assert ci.upper == 0.0
    
    def test_bootstrap_ci_single_value(self):
        """Test bootstrap CI with single value."""
        data = [0.75]
        ci = ConfidenceEstimator.bootstrap_ci(data)
        
        assert ci.point_estimate == 0.75
        assert ci.lower <= 0.75 <= ci.upper
    
    def test_wilson_ci_basic(self):
        """Test Wilson score confidence interval."""
        ci = ConfidenceEstimator.wilson_ci(successes=80, trials=100, ci=0.95)
        
        assert ci.point_estimate == 0.8
        assert ci.lower < 0.8
        assert ci.upper > 0.8
        assert ci.lower >= 0.0
        assert ci.upper <= 1.0
    
    def test_wilson_ci_zero_trials(self):
        """Test Wilson CI with zero trials."""
        ci = ConfidenceEstimator.wilson_ci(0, 0)
        
        assert ci.point_estimate == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0
    
    def test_wilson_ci_perfect_success(self):
        """Test Wilson CI with perfect success."""
        ci = ConfidenceEstimator.wilson_ci(10, 10)
        
        assert ci.point_estimate == 1.0
        assert ci.lower > 0.5  # Should have reasonable lower bound
        assert ci.upper == 1.0
    
    def test_wilson_ci_bounds(self):
        """Test Wilson CI stays within [0, 1]."""
        for successes in [0, 5, 10, 15, 20]:
            for trials in [20, 50, 100]:
                if successes <= trials:
                    ci = ConfidenceEstimator.wilson_ci(successes, trials)
                    assert 0 <= ci.lower <= 1
                    assert 0 <= ci.upper <= 1


class TestMasteryLevel:
    """Test suite for mastery level classification."""
    
    def test_classify_novice(self):
        """Test novice classification."""
        level = MasteryLevel.classify(0.2)
        assert level == 'novice'
    
    def test_classify_intermediate(self):
        """Test intermediate classification."""
        level = MasteryLevel.classify(0.5)
        assert level == 'intermediate'
    
    def test_classify_expert(self):
        """Test expert classification."""
        level = MasteryLevel.classify(0.8)
        assert level == 'expert'
    
    def test_classify_boundaries(self):
        """Test classification at boundaries."""
        # At novice boundary
        level = MasteryLevel.classify(0.3)
        assert level == 'intermediate'
        
        # At intermediate boundary
        level = MasteryLevel.classify(0.7)
        assert level == 'intermediate'
    
    def test_classify_custom_thresholds(self):
        """Test classification with custom thresholds."""
        level = MasteryLevel.classify(0.6, novice_threshold=0.5,
                                     intermediate_threshold=0.8)
        assert level == 'intermediate'


class TestMasteryTimeline:
    """Test suite for mastery timeline tracking."""
    
    def test_initialization(self):
        """Test timeline initialization."""
        timeline = MasteryTimeline()
        assert timeline.snapshots == []
        assert timeline.learning_events == []
    
    def test_add_snapshot(self):
        """Test adding mastery snapshot."""
        timeline = MasteryTimeline()
        
        snapshot = MasterySnapshot(
            timestamp=datetime.now(),
            p_knowledge=0.5,
            n_attempts=10,
            n_correct=5,
            accuracy=0.5
        )
        
        timeline.add_snapshot(snapshot)
        assert len(timeline.snapshots) == 1
        assert timeline.snapshots[0].p_knowledge == 0.5
    
    def test_learning_velocity(self):
        """Test learning velocity calculation."""
        timeline = MasteryTimeline()
        
        now = datetime.now()
        
        # Add two snapshots with known mastery change
        snapshot1 = MasterySnapshot(
            timestamp=now - timedelta(hours=1),
            p_knowledge=0.5,
            n_attempts=5,
            n_correct=2,
            accuracy=0.4
        )
        
        snapshot2 = MasterySnapshot(
            timestamp=now,
            p_knowledge=0.6,
            n_attempts=10,
            n_correct=6,
            accuracy=0.6
        )
        
        timeline.add_snapshot(snapshot1)
        timeline.add_snapshot(snapshot2)
        
        velocity = timeline.get_learning_velocity(window_hours=2)
        assert velocity is not None
        assert abs(velocity - 0.1) < 0.01  # 0.1 mastery per hour
    
    def test_time_to_mastery_estimation(self):
        """Test time to mastery estimation."""
        timeline = MasteryTimeline()
        
        now = datetime.now()
        
        # Create progression towards mastery
        for i in range(5):
            snapshot = MasterySnapshot(
                timestamp=now - timedelta(hours=5-i),
                p_knowledge=0.5 + i * 0.08,
                n_attempts=(i+1) * 5,
                n_correct=(i+1) * 3,
                accuracy=0.6
            )
            timeline.add_snapshot(snapshot)
        
        time_to_mastery = timeline.estimate_time_to_mastery(target=0.9)
        
        # Should estimate some time needed
        if time_to_mastery:
            assert time_to_mastery.total_seconds() > 0
    
    def test_time_to_mastery_already_mastered(self):
        """Test time to mastery when already mastered."""
        timeline = MasteryTimeline()
        
        snapshot = MasterySnapshot(
            timestamp=datetime.now(),
            p_knowledge=0.95,
            n_attempts=20,
            n_correct=19,
            accuracy=0.95
        )
        
        timeline.add_snapshot(snapshot)
        
        time_to_mastery = timeline.estimate_time_to_mastery(target=0.9)
        assert time_to_mastery == timedelta(0)
    
    def test_get_progression(self):
        """Test getting progression history."""
        timeline = MasteryTimeline()
        
        now = datetime.now()
        for i in range(3):
            snapshot = MasterySnapshot(
                timestamp=now + timedelta(hours=i),
                p_knowledge=0.3 + i * 0.2,
                n_attempts=5*(i+1),
                n_correct=int(3*(i+1)),
                accuracy=0.6
            )
            timeline.add_snapshot(snapshot)
        
        progression = timeline.get_progression()
        assert len(progression) == 3
        
        # Check progression is monotonic or reasonable
        for t, p in progression:
            assert isinstance(t, datetime)
            assert 0 <= p <= 1
    
    def test_get_summary(self):
        """Test getting timeline summary."""
        timeline = MasteryTimeline()
        
        # Empty timeline
        summary = timeline.get_summary()
        assert summary['total_snapshots'] == 0
        
        # Add snapshot
        snapshot = MasterySnapshot(
            timestamp=datetime.now(),
            p_knowledge=0.7,
            n_attempts=10,
            n_correct=7,
            accuracy=0.7
        )
        timeline.add_snapshot(snapshot)
        
        summary = timeline.get_summary()
        assert summary['total_snapshots'] == 1
        assert summary['current_mastery'] == 0.7
        assert summary['average_mastery'] == 0.7


class TestMasteryEstimator:
    """Test suite for main mastery estimator."""
    
    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        estimator = MasteryEstimator()
        concept = Concept(name='algebra')
        knowledge_state = KnowledgeState(concept=concept)
        return estimator, concept, knowledge_state
    
    def test_initialization(self, setup):
        """Test estimator initialization."""
        estimator, _, _ = setup
        
        assert estimator.mastery_threshold == 0.9
        assert estimator.novice_threshold == 0.3
        assert estimator.intermediate_threshold == 0.7
        assert isinstance(estimator.bkt, BayesianKnowledgeTracer)
        assert isinstance(estimator.irt, IRTEstimator)
    
    def test_custom_thresholds(self):
        """Test estimator with custom thresholds."""
        estimator = MasteryEstimator(
            mastery_threshold=0.85,
            novice_threshold=0.25,
            intermediate_threshold=0.65
        )
        
        assert estimator.mastery_threshold == 0.85
        assert estimator.novice_threshold == 0.25
        assert estimator.intermediate_threshold == 0.65
    
    def test_update_from_response_correct(self, setup):
        """Test updating from correct response."""
        estimator, _, knowledge_state = setup
        
        response = InteractionResponse(is_correct=True)
        updated_state = estimator.update_from_response(knowledge_state, response)
        
        # Knowledge should increase
        assert updated_state.p_knowledge > knowledge_state.p_knowledge
        assert updated_state.n_attempts == 1
        assert updated_state.n_correct == 1
    
    def test_update_from_response_incorrect(self, setup):
        """Test updating from incorrect response."""
        estimator, _, knowledge_state = setup
        
        # Set initial high knowledge
        knowledge_state.p_knowledge = 0.8
        
        response = InteractionResponse(is_correct=False)
        updated_state = estimator.update_from_response(knowledge_state, response)
        
        # Knowledge should decrease
        assert updated_state.p_knowledge < knowledge_state.p_knowledge
        assert updated_state.n_attempts == 1
        assert updated_state.n_correct == 0
    
    def test_multiple_updates(self, setup):
        """Test multiple sequential updates."""
        estimator, _, knowledge_state = setup
        
        responses = [True, True, False, True, True]
        
        for is_correct in responses:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        assert knowledge_state.n_attempts == 5
        assert knowledge_state.n_correct == 4
    
    def test_get_mastery_estimate(self, setup):
        """Test getting mastery estimate."""
        estimator, _, knowledge_state = setup
        
        # Update with responses
        for is_correct in [True, True, False, True]:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        estimate = estimator.get_mastery_estimate('algebra', knowledge_state)
        
        assert 'bkt_mastery' in estimate
        assert 'irt_mastery' in estimate
        assert 'average_mastery' in estimate
        assert 'confidence_interval' in estimate
        assert 'mastery_level' in estimate
        assert 'is_mastered' in estimate
        assert 'attempts' in estimate
        assert 'correct' in estimate
        assert 'accuracy' in estimate
    
    def test_mastery_classification(self, setup):
        """Test mastery classification."""
        estimator, _, knowledge_state = setup
        
        # Low mastery
        knowledge_state.p_knowledge = 0.2
        estimate = estimator.get_mastery_estimate('algebra', knowledge_state)
        assert estimate['mastery_level'] == 'novice'
        assert not estimate['is_mastered']
        
        # Medium mastery
        knowledge_state.p_knowledge = 0.5
        estimate = estimator.get_mastery_estimate('algebra', knowledge_state)
        assert estimate['mastery_level'] == 'intermediate'
        
        # High mastery
        knowledge_state.p_knowledge = 0.92
        estimate = estimator.get_mastery_estimate('algebra', knowledge_state)
        assert estimate['mastery_level'] == 'expert'
        assert estimate['is_mastered']
    
    def test_set_irt_parameters(self, setup):
        """Test setting IRT parameters."""
        estimator, _, _ = setup
        
        estimator.set_irt_parameters('algebra', difficulty=0.3,
                                    discrimination=1.5, guessing=0.15)
        
        params = estimator.irt.get_concept_parameters('algebra')
        assert params['difficulty'] == 0.3
        assert params['discrimination'] == 1.5
        assert params['guessing'] == 0.15
    
    def test_get_mastery_timeline(self, setup):
        """Test getting mastery timeline."""
        estimator, _, knowledge_state = setup
        
        # Initially no timeline
        timeline = estimator.get_mastery_timeline('algebra')
        assert timeline is None
        
        # Add responses to create timeline
        for is_correct in [True, True, False, True]:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        timeline = estimator.get_mastery_timeline('algebra')
        assert timeline is not None
        assert 'total_snapshots' in timeline
        assert timeline['total_snapshots'] > 0
    
    def test_compare_bkt_vs_irt(self, setup):
        """Test BKT vs IRT comparison."""
        estimator, _, knowledge_state = setup
        
        # Add responses
        for is_correct in [True, True, False, True, True]:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        comparison = estimator.compare_bkt_vs_irt('algebra', knowledge_state)
        
        assert 'bkt' in comparison
        assert 'irt' in comparison
        assert 'difference' in comparison
        assert 'agreement' in comparison
        assert 'estimate' in comparison


class TestIntegration:
    """Integration tests for the complete mastery estimation system."""
    
    def test_full_workflow(self):
        """Test complete workflow from responses to mastery estimate."""
        estimator = MasteryEstimator()
        concept = Concept(name='calculus')
        knowledge_state = KnowledgeState(concept=concept)
        
        # Simulate learning sequence
        responses_sequence = [False, False, True, True, False, True, True, True]
        
        for is_correct in responses_sequence:
            response = InteractionResponse(is_correct=is_correct)
            knowledge_state = estimator.update_from_response(knowledge_state, response)
        
        # Get comprehensive estimate
        estimate = estimator.get_mastery_estimate('calculus', knowledge_state)
        
        # Verify estimate quality
        assert estimate['attempts'] == len(responses_sequence)
        assert estimate['correct'] == sum(responses_sequence)
        assert estimate['bkt_mastery'] > 0.5  # Should have learned
        assert 0 <= estimate['accuracy'] <= 1
        
        # Check timeline
        timeline = estimator.get_mastery_timeline('calculus')
        assert timeline is not None
        assert timeline['current_mastery'] == estimate['bkt_mastery']
    
    def test_multiple_concepts(self):
        """Test tracking multiple concepts simultaneously."""
        estimator = MasteryEstimator()
        
        concepts = {
            'algebra': Concept(name='algebra'),
            'geometry': Concept(name='geometry'),
            'calculus': Concept(name='calculus')
        }
        
        knowledge_states = {
            name: KnowledgeState(concept=concept)
            for name, concept in concepts.items()
        }
        
        # Different learning patterns for each concept
        patterns = {
            'algebra': [True, True, True, True],  # Fast learner
            'geometry': [False, True, False, True],  # Struggling
            'calculus': [True, False, True, False]  # Inconsistent
        }
        
        for concept_name, responses in patterns.items():
            for is_correct in responses:
                response = InteractionResponse(is_correct=is_correct)
                knowledge_states[concept_name] = estimator.update_from_response(
                    knowledge_states[concept_name], response
                )
        
        # Get estimates for all concepts
        estimates = {
            name: estimator.get_mastery_estimate(name, state)
            for name, state in knowledge_states.items()
        }
        
        # Verify differentiation
        algebra_mastery = estimates['algebra']['bkt_mastery']
        geometry_mastery = estimates['geometry']['bkt_mastery']
        
        assert algebra_mastery > geometry_mastery  # Algebra should be higher
