"""
Knowledge Mastery Estimation Module

Implements Bayesian Knowledge Tracing (BKT) and Item Response Theory (IRT)
for estimating knowledge mastery with confidence intervals and timeline tracking.

Key Components:
- BayesianKnowledgeTracer: BKT algorithm implementation
- IRTEstimator: 3-parameter IRT model for mastery estimation
- MasteryEstimator: Main estimator combining BKT and IRT
- MasteryTimeline: Tracks mastery progression over time
- ConfidenceEstimator: Bootstrap-based uncertainty quantification
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from src.core.user.models.knowledge_state import KnowledgeState
from src.core.knowledge.models.concept import Concept


class ConfidenceInterval(NamedTuple):
    """Represents a confidence interval estimate."""
    lower: float
    point_estimate: float
    upper: float
    confidence_level: float = 0.95


@dataclass
class MasteryLevel:
    """Represents mastery level classification."""
    level: str  # 'novice', 'intermediate', 'expert'
    p_mastery: float  # Probability of mastery (0-1)
    confidence_interval: Optional[ConfidenceInterval] = None
    
    @staticmethod
    def classify(p_knowledge: float, novice_threshold: float = 0.3,
                 intermediate_threshold: float = 0.7) -> str:
        """Classify knowledge level based on p_knowledge."""
        if p_knowledge < novice_threshold:
            return 'novice'
        elif p_knowledge < intermediate_threshold:
            return 'intermediate'
        else:
            return 'expert'


@dataclass
class InteractionResponse:
    """Represents a user's response to an interaction."""
    is_correct: bool
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_seconds: Optional[float] = None
    confidence: Optional[float] = None  # User's confidence in response


class BayesianKnowledgeTracer:
    """
    Implements Bayesian Knowledge Tracing (BKT) algorithm.
    
    BKT tracks the probability of a student's knowledge based on their
    observed responses. Key parameters:
    - p_knowledge: Probability student knows the concept
    - p_learn: Probability of learning from an attempt
    - p_guess: Probability of correct guess without knowledge
    - p_slip: Probability of slipping (error despite knowing)
    
    Model:
        P(correct) = p_knowledge × (1 - p_slip) + (1 - p_knowledge) × p_guess
        
    Updates on correct response:
        P(L_t) = P(L_{t-1}) + (1 - P(L_{t-1})) × p_learn
        
    Updates on incorrect response:
        P(L_t) = P(L_{t-1}) × (1 - p_slip)
    """
    
    def __init__(self, p_learn: float = 0.3, p_guess: float = 0.2,
                 p_slip: float = 0.1):
        """
        Initialize BKT tracer.
        
        Args:
            p_learn: Learning probability (default 0.3)
            p_guess: Guess probability (default 0.2)
            p_slip: Slip probability (default 0.1)
        """
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip
    
    def observation_probability(self, p_knowledge: float,
                               is_correct: bool) -> float:
        """
        Calculate probability of observed response.
        
        P(correct | L) = P(L) × (1 - p_slip) + (1 - P(L)) × p_guess
        P(incorrect | L) = P(L) × p_slip + (1 - P(L)) × (1 - p_guess)
        
        Args:
            p_knowledge: Current knowledge probability
            is_correct: Whether response was correct
            
        Returns:
            Probability of observation
        """
        if is_correct:
            return p_knowledge * (1 - self.p_slip) + (1 - p_knowledge) * self.p_guess
        else:
            return p_knowledge * self.p_slip + (1 - p_knowledge) * (1 - self.p_guess)
    
    def update_knowledge(self, p_knowledge: float,
                        is_correct: bool) -> float:
        """
        Update knowledge probability using Bayesian update rule.
        
        Uses Bayes' rule: P(L_t | o_t) = P(o_t | L_t) × P(L_t) / P(o_t)
        
        Args:
            p_knowledge: Current knowledge probability
            is_correct: Whether response was correct
            
        Returns:
            Updated knowledge probability
        """
        # Observation probability given knowledge
        p_obs_given_know = self.observation_probability(p_knowledge, is_correct)
        
        # Prior probability of knowledge (accounting for learning)
        p_know_prior = p_knowledge + (1 - p_knowledge) * self.p_learn
        
        # Total observation probability
        p_obs = self.observation_probability(p_know_prior, is_correct)
        
        if p_obs == 0:
            return p_know_prior
        
        # Posterior knowledge probability
        p_know_posterior = (self.observation_probability(p_know_prior, is_correct) * 
                           p_know_prior) / p_obs
        
        return np.clip(p_know_posterior, 0, 1)
    
    def trace_sequence(self, initial_knowledge: float,
                      responses: List[bool]) -> Tuple[List[float], float]:
        """
        Trace through a sequence of responses.
        
        Args:
            initial_knowledge: Starting knowledge probability
            responses: List of True (correct) or False (incorrect)
            
        Returns:
            Tuple of (knowledge_trajectory, final_knowledge)
        """
        trajectory = [initial_knowledge]
        current_knowledge = initial_knowledge
        
        for is_correct in responses:
            current_knowledge = self.update_knowledge(current_knowledge, is_correct)
            trajectory.append(current_knowledge)
        
        return trajectory, current_knowledge


class IRTEstimator:
    """
    Implements Item Response Theory (IRT) for mastery estimation.
    
    Uses 3-parameter logistic (3PL) model:
        P(correct | θ, α, β, γ) = γ + (1 - γ) × Λ(α × (θ - β))
    
    Where:
    - θ: Student ability
    - α: Item discrimination
    - β: Item difficulty
    - γ: Guessing parameter
    - Λ(x): Logistic function = 1 / (1 + e^(-x))
    """
    
    def __init__(self):
        """Initialize IRT estimator."""
        self.concept_params: Dict[str, Dict[str, float]] = {}
    
    def logistic(self, x: float) -> float:
        """Logistic function: 1 / (1 + e^(-x))."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def response_probability(self, ability: float, discrimination: float,
                            difficulty: float, guessing: float) -> float:
        """
        Calculate probability of correct response using 3PL model.
        
        P(correct) = γ + (1 - γ) × Λ(α × (θ - β))
        
        Args:
            ability: Student ability (θ)
            discrimination: Item discrimination (α)
            difficulty: Item difficulty (β)
            guessing: Guessing parameter (γ)
            
        Returns:
            Probability of correct response
        """
        exponent = discrimination * (ability - difficulty)
        logistic_prob = self.logistic(exponent)
        return guessing + (1 - guessing) * logistic_prob
    
    def estimate_ability_mle(self, responses: List[bool],
                            difficulty: float = 0.0,
                            discrimination: float = 1.0,
                            guessing: float = 0.2,
                            max_iterations: int = 50) -> float:
        """
        Estimate student ability using Maximum Likelihood Estimation.
        
        Args:
            responses: List of True (correct) or False (incorrect)
            difficulty: Item difficulty parameter
            discrimination: Item discrimination parameter
            guessing: Guessing parameter
            max_iterations: Maximum iterations for optimization
            
        Returns:
            Estimated ability level
        """
        if not responses:
            return 0.0
        
        # Initial ability estimate at 0
        ability = 0.0
        learning_rate = 0.1
        
        for _ in range(max_iterations):
            # Calculate log-likelihood gradient
            gradient = 0.0
            
            for is_correct in responses:
                prob = self.response_probability(ability, discrimination,
                                                difficulty, guessing)
                
                if is_correct:
                    gradient += discrimination * (1 - guessing) * (1 - prob) / \
                               (prob + 1e-10)
                else:
                    gradient -= discrimination * (1 - guessing) * prob / \
                               ((1 - prob) + 1e-10)
            
            # Update ability
            new_ability = ability + learning_rate * gradient / len(responses)
            
            # Check convergence
            if abs(new_ability - ability) < 1e-6:
                break
            
            ability = new_ability
            learning_rate *= 0.99  # Decay learning rate
        
        return np.clip(ability, -3, 3)
    
    def set_concept_parameters(self, concept: str, difficulty: float,
                               discrimination: float, guessing: float = 0.2):
        """
        Set IRT parameters for a concept.
        
        Args:
            concept: Concept name
            difficulty: Difficulty parameter
            discrimination: Discrimination parameter
            guessing: Guessing parameter
        """
        self.concept_params[concept] = {
            'difficulty': difficulty,
            'discrimination': discrimination,
            'guessing': guessing
        }
    
    def get_concept_parameters(self, concept: str) -> Dict[str, float]:
        """Get IRT parameters for a concept."""
        return self.concept_params.get(concept, {
            'difficulty': 0.0,
            'discrimination': 1.0,
            'guessing': 0.2
        })


class ConfidenceEstimator:
    """
    Bootstrap-based uncertainty quantification for mastery estimates.
    
    Provides confidence intervals around point estimates using
    Monte Carlo bootstrap resampling.
    """
    
    @staticmethod
    def bootstrap_ci(data: List[float], ci: float = 0.95,
                    n_bootstrap: int = 1000) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Input data samples
            ci: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            ConfidenceInterval with lower, point, upper estimates
        """
        if not data:
            return ConfidenceInterval(0.0, 0.0, 0.0, ci)
        
        point_estimate = np.mean(data)
        
        # Generate bootstrap samples
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate percentile CI
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(float(lower), float(point_estimate), float(upper), float(ci))
    
    @staticmethod
    def wilson_ci(successes: int, trials: int, ci: float = 0.95) -> ConfidenceInterval:
        """
        Calculate Wilson score confidence interval for binary outcomes.
        
        More accurate than normal approximation for small samples and
        extreme proportions.
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            ci: Confidence level
            
        Returns:
            ConfidenceInterval
        """
        if trials == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0, ci)
        
        p_hat = successes / trials
        z = stats.norm.ppf((1 + ci) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
        
        lower = np.clip(center - margin, 0, 1)
        upper = np.clip(center + margin, 0, 1)
        
        return ConfidenceInterval(float(lower), float(p_hat), float(upper), float(ci))


@dataclass
class MasterySnapshot:
    """Snapshot of mastery state at a point in time."""
    timestamp: datetime
    p_knowledge: float
    p_knowledge_irt: Optional[float] = None
    n_attempts: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    confidence_interval: Optional[ConfidenceInterval] = None
    mastery_level: Optional[str] = None


class MasteryTimeline:
    """
    Tracks mastery progression over time.
    
    Maintains snapshots of mastery estimates, identifies learning patterns,
    calculates learning velocity, and estimates time-to-mastery.
    """
    
    def __init__(self):
        """Initialize mastery timeline."""
        self.snapshots: List[MasterySnapshot] = []
        self.learning_events: List[Dict] = []  # Track spikes/plateaus
    
    def add_snapshot(self, snapshot: MasterySnapshot):
        """Add a mastery snapshot."""
        self.snapshots.append(snapshot)
        self._detect_learning_events()
    
    def _detect_learning_events(self):
        """Detect learning spikes and plateaus."""
        if len(self.snapshots) < 2:
            return
        
        recent_snapshots = self.snapshots[-5:]  # Look at last 5
        
        if len(recent_snapshots) < 2:
            return
        
        # Calculate velocity (change in mastery)
        velocities = []
        for i in range(1, len(recent_snapshots)):
            time_delta = (recent_snapshots[i].timestamp - 
                         recent_snapshots[i-1].timestamp).total_seconds() / 3600
            if time_delta > 0:
                mastery_delta = (recent_snapshots[i].p_knowledge - 
                                recent_snapshots[i-1].p_knowledge)
                velocity = mastery_delta / time_delta
                velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            
            # Detect spike (high positive velocity)
            if avg_velocity > 0.05:
                self.learning_events.append({
                    'type': 'spike',
                    'timestamp': self.snapshots[-1].timestamp,
                    'velocity': avg_velocity
                })
            
            # Detect plateau (low velocity)
            elif avg_velocity < 0.01 and self.snapshots[-1].p_knowledge > 0.5:
                self.learning_events.append({
                    'type': 'plateau',
                    'timestamp': self.snapshots[-1].timestamp,
                    'velocity': avg_velocity
                })
    
    def get_learning_velocity(self, window_hours: int = 24) -> Optional[float]:
        """
        Calculate learning velocity over a time window.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Learning velocity (mastery change per hour) or None
        """
        if len(self.snapshots) < 2:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        relevant_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(relevant_snapshots) < 2:
            return None
        
        time_span = (relevant_snapshots[-1].timestamp - 
                    relevant_snapshots[0].timestamp).total_seconds() / 3600
        
        if time_span == 0:
            return None
        
        mastery_change = relevant_snapshots[-1].p_knowledge - relevant_snapshots[0].p_knowledge
        return mastery_change / time_span
    
    def estimate_time_to_mastery(self, target: float = 0.9) -> Optional[timedelta]:
        """
        Estimate time to reach mastery threshold.
        
        Args:
            target: Target mastery level
            
        Returns:
            Estimated time to mastery or None
        """
        if not self.snapshots:
            return None
        
        current_mastery = self.snapshots[-1].p_knowledge
        
        if current_mastery >= target:
            return timedelta(0)  # Already at target
        
        velocity = self.get_learning_velocity(window_hours=72)  # Use 3-day window
        
        if velocity is None or velocity <= 0:
            return None  # Cannot estimate with no progress
        
        mastery_needed = target - current_mastery
        hours_needed = mastery_needed / velocity
        
        return timedelta(hours=hours_needed)
    
    def get_progression(self) -> List[Tuple[datetime, float]]:
        """Get mastery progression as list of (timestamp, p_knowledge)."""
        return [(s.timestamp, s.p_knowledge) for s in self.snapshots]
    
    def get_summary(self) -> Dict:
        """Get timeline summary."""
        if not self.snapshots:
            return {
                'total_snapshots': 0,
                'current_mastery': None,
                'average_mastery': None,
                'max_mastery': None,
                'learning_events': []
            }
        
        mastery_values = [s.p_knowledge for s in self.snapshots]
        
        return {
            'total_snapshots': len(self.snapshots),
            'current_mastery': self.snapshots[-1].p_knowledge,
            'average_mastery': np.mean(mastery_values),
            'max_mastery': np.max(mastery_values),
            'min_mastery': np.min(mastery_values),
            'learning_velocity': self.get_learning_velocity(),
            'time_to_mastery': self.estimate_time_to_mastery(),
            'learning_events': self.learning_events
        }


class MasteryEstimator:
    """
    Main mastery estimator combining BKT and IRT approaches.
    
    Provides unified interface for mastery estimation with confidence
    intervals and timeline tracking.
    """
    
    def __init__(self, mastery_threshold: float = 0.9,
                 novice_threshold: float = 0.3,
                 intermediate_threshold: float = 0.7):
        """
        Initialize mastery estimator.
        
        Args:
            mastery_threshold: Threshold for mastery (default 0.9)
            novice_threshold: Threshold for novice level (default 0.3)
            intermediate_threshold: Threshold for intermediate (default 0.7)
        """
        self.mastery_threshold = mastery_threshold
        self.novice_threshold = novice_threshold
        self.intermediate_threshold = intermediate_threshold
        
        self.bkt = BayesianKnowledgeTracer()
        self.irt = IRTEstimator()
        self.confidence_estimator = ConfidenceEstimator()
        
        self.timelines: Dict[str, MasteryTimeline] = {}  # concept -> timeline
        self.response_history: Dict[str, List[bool]] = {}  # concept -> responses
    
    def update_from_response(self, knowledge_state: KnowledgeState,
                            response: InteractionResponse) -> KnowledgeState:
        """
        Update knowledge state based on response using BKT.
        
        Args:
            knowledge_state: Current knowledge state
            response: User response
            
        Returns:
            Updated knowledge state
        """
        concept_name = knowledge_state.concept.name
        
        # Update BKT knowledge
        new_p_knowledge = self.bkt.update_knowledge(
            knowledge_state.p_knowledge,
            response.is_correct
        )
        
        # Update learning parameters if we have enough data
        if knowledge_state.n_attempts > 0:
            accuracy = knowledge_state.n_correct / knowledge_state.n_attempts
            
            # Adaptively update p_learn based on observed learning rate
            if accuracy > 0.7:
                self.bkt.p_learn = min(0.5, self.bkt.p_learn + 0.02)
            elif accuracy < 0.3:
                self.bkt.p_learn = max(0.1, self.bkt.p_learn - 0.02)
        
        # Update knowledge state
        knowledge_state.p_knowledge = new_p_knowledge
        knowledge_state.n_attempts += 1
        if response.is_correct:
            knowledge_state.n_correct += 1
        knowledge_state.last_interaction = response.timestamp
        
        # Track response history
        if concept_name not in self.response_history:
            self.response_history[concept_name] = []
        self.response_history[concept_name].append(response.is_correct)
        
        # Update timeline
        self._update_timeline(concept_name, knowledge_state, response)
        
        return knowledge_state
    
    def _update_timeline(self, concept_name: str, knowledge_state: KnowledgeState,
                        response: InteractionResponse):
        """Update mastery timeline for concept."""
        if concept_name not in self.timelines:
            self.timelines[concept_name] = MasteryTimeline()
        
        # Calculate IRT estimate if parameters exist
        p_knowledge_irt = None
        irt_params = self.irt.get_concept_parameters(concept_name)
        responses = self.response_history.get(concept_name, [])
        if responses:
            ability = self.irt.estimate_ability_mle(
                responses,
                irt_params['difficulty'],
                irt_params['discrimination'],
                irt_params['guessing']
            )
            # Convert ability (-3 to 3 scale) to mastery probability (0 to 1)
            p_knowledge_irt = (ability + 3) / 6
        
        # Create confidence interval
        if knowledge_state.n_attempts > 0:
            ci = self.confidence_estimator.wilson_ci(
                knowledge_state.n_correct,
                knowledge_state.n_attempts
            )
        else:
            ci = None
        
        # Determine mastery level
        mastery_level = MasteryLevel.classify(
            knowledge_state.p_knowledge,
            self.novice_threshold,
            self.intermediate_threshold
        )
        
        # Create snapshot
        snapshot = MasterySnapshot(
            timestamp=response.timestamp,
            p_knowledge=knowledge_state.p_knowledge,
            p_knowledge_irt=p_knowledge_irt,
            n_attempts=knowledge_state.n_attempts,
            n_correct=knowledge_state.n_correct,
            accuracy=knowledge_state.n_correct / knowledge_state.n_attempts
                    if knowledge_state.n_attempts > 0 else 0.0,
            confidence_interval=ci,
            mastery_level=mastery_level
        )
        
        self.timelines[concept_name].add_snapshot(snapshot)
    
    def get_mastery_estimate(self, concept_name: str,
                            knowledge_state: KnowledgeState) -> Dict:
        """
        Get comprehensive mastery estimate for a concept.
        
        Args:
            concept_name: Name of concept
            knowledge_state: Knowledge state
            
        Returns:
            Dictionary with mastery estimates and confidence
        """
        responses = self.response_history.get(concept_name, [])
        
        # BKT estimate
        bkt_mastery = knowledge_state.p_knowledge
        
        # IRT estimate
        irt_params = self.irt.get_concept_parameters(concept_name)
        irt_mastery = None
        if responses:
            ability = self.irt.estimate_ability_mle(
                responses,
                irt_params['difficulty'],
                irt_params['discrimination'],
                irt_params['guessing']
            )
            irt_mastery = (ability + 3) / 6
        
        # Confidence interval
        ci = None
        if knowledge_state.n_attempts > 0:
            ci = self.confidence_estimator.wilson_ci(
                knowledge_state.n_correct,
                knowledge_state.n_attempts
            )
        
        # Mastery classification
        mastery_level = MasteryLevel.classify(
            bkt_mastery,
            self.novice_threshold,
            self.intermediate_threshold
        )
        
        # Is mastered?
        is_mastered = bkt_mastery >= self.mastery_threshold
        
        return {
            'concept': concept_name,
            'bkt_mastery': bkt_mastery,
            'irt_mastery': irt_mastery,
            'average_mastery': np.mean([m for m in [bkt_mastery, irt_mastery]
                                       if m is not None]),
            'confidence_interval': ci,
            'mastery_level': mastery_level,
            'is_mastered': is_mastered,
            'attempts': knowledge_state.n_attempts,
            'correct': knowledge_state.n_correct,
            'accuracy': knowledge_state.n_correct / knowledge_state.n_attempts
                       if knowledge_state.n_attempts > 0 else 0.0
        }
    
    def get_mastery_timeline(self, concept_name: str) -> Optional[Dict]:
        """
        Get mastery timeline for a concept.
        
        Args:
            concept_name: Name of concept
            
        Returns:
            Timeline dictionary or None
        """
        if concept_name not in self.timelines:
            return None
        
        timeline = self.timelines[concept_name]
        return timeline.get_summary()
    
    def set_irt_parameters(self, concept_name: str, difficulty: float,
                          discrimination: float, guessing: float = 0.2):
        """
        Set IRT parameters for a concept.
        
        Args:
            concept_name: Concept name
            difficulty: Difficulty parameter
            discrimination: Discrimination parameter
            guessing: Guessing parameter
        """
        self.irt.set_concept_parameters(concept_name, difficulty,
                                       discrimination, guessing)
    
    def compare_bkt_vs_irt(self, concept_name: str,
                          knowledge_state: KnowledgeState) -> Dict:
        """
        Compare BKT and IRT mastery estimates.
        
        Args:
            concept_name: Concept name
            knowledge_state: Knowledge state
            
        Returns:
            Comparison dictionary
        """
        estimate = self.get_mastery_estimate(concept_name, knowledge_state)
        
        bkt = estimate['bkt_mastery']
        irt = estimate['irt_mastery']
        
        if irt is None:
            return {
                'bkt': bkt,
                'irt': irt,
                'difference': None,
                'agreement': None
            }
        
        difference = abs(bkt - irt)
        agreement = 1 - difference  # Higher is better agreement
        
        return {
            'bkt': bkt,
            'irt': irt,
            'difference': difference,
            'agreement': agreement,
            'estimate': estimate['average_mastery']
        }
