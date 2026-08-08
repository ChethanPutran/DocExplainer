"""
Curriculum Generator Module

Implements prerequisite-ordered learning paths with adaptive sequencing,
multiple curriculum strategies, and concept dependency resolution.

Key Features:
- Prerequisite-ordered learning paths respecting concept dependencies
- Adaptive sequencing based on user progress and mastery levels
- Multiple curriculum strategies (breadth-first, depth-first, adaptive, etc.)
- Circular dependency detection and handling
- Support for partial learning paths
- Time-to-completion estimation
- Strategy selection based on user preferences
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import networkx as nx
from collections import deque

from src.core.orchestrator.models.curriculum_models import (
    CurriculumStrategy,
    CurriculumNode,
    LearningPath,
    PathProgressState,
)
from src.core.knowledge.models.graph import ConceptGraph
from src.core.user.services.user_profile_service import UserProfileService

logger = logging.getLogger(__name__)


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected in concept graph."""
    pass


class ConceptDependencyResolver:
    """Resolves concept dependencies in a directed graph."""
    
    def __init__(self, concept_graph: ConceptGraph):
        """
        Initialize the resolver.
        
        Args:
            concept_graph: The concept graph to resolve dependencies from.
        """
        self.concept_graph = concept_graph
        self.dependency_cache: Dict[str, Set[str]] = {}
        self.depth_cache: Dict[str, int] = {}
        self._has_cycles = None
    
    def get_direct_dependencies(self, concept_id: str) -> Set[str]:
        """
        Get direct prerequisites for a concept.
        
        Args:
            concept_id: The concept ID to get dependencies for.
        
        Returns:
            Set of direct prerequisite concept IDs.
        """
        if not self.concept_graph.has_concept(concept_id):
            return set()
        
        predecessors = set(self.concept_graph.graph.predecessors(concept_id))
        return predecessors
    
    def get_transitive_dependencies(self, concept_id: str) -> Set[str]:
        """
        Get all transitive dependencies (prerequisites and their prerequisites).
        
        Args:
            concept_id: The concept ID to get dependencies for.
        
        Returns:
            Set of all prerequisite concept IDs (direct and indirect).
        
        Raises:
            CircularDependencyError: If circular dependencies are detected.
        """
        if concept_id in self.dependency_cache:
            return self.dependency_cache[concept_id].copy()
        
        visited = set()
        all_deps = set()
        stack = [concept_id]
        
        while stack:
            current = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            direct_deps = self.get_direct_dependencies(current)
            all_deps.update(direct_deps)
            stack.extend(direct_deps)
        
        # Check for cycles: if concept is in its own dependencies
        if concept_id in all_deps:
            raise CircularDependencyError(f"Circular dependency detected for {concept_id}")
        
        self.dependency_cache[concept_id] = all_deps
        return all_deps.copy()
    
    def get_dependency_depth(self, concept_id: str) -> int:
        """
        Get the depth of prerequisites (max chain length).
        
        Args:
            concept_id: The concept ID to get depth for.
        
        Returns:
            The maximum depth of prerequisites (0 if no prerequisites).
        """
        if concept_id in self.depth_cache:
            return self.depth_cache[concept_id]
        
        direct_deps = self.get_direct_dependencies(concept_id)
        
        if not direct_deps:
            depth = 0
        else:
            depth = 1 + max(self.get_dependency_depth(dep) for dep in direct_deps)
        
        self.depth_cache[concept_id] = depth
        return depth
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect all cycles in the graph.
        
        Returns:
            List of cycles, where each cycle is a list of concept IDs.
        """
        try:
            cycles = list(nx.simple_cycles(self.concept_graph.graph))
            return cycles
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return []
    
    def has_cycles(self) -> bool:
        """Check if the graph has any cycles."""
        if self._has_cycles is None:
            self._has_cycles = len(self.detect_cycles()) > 0
        return self._has_cycles
    
    def topological_sort(self) -> List[str]:
        """
        Get topological sort of concepts respecting dependencies.
        
        Returns:
            List of concept IDs in topological order.
        
        Raises:
            CircularDependencyError: If the graph has cycles.
        """
        if self.has_cycles():
            cycles = self.detect_cycles()
            raise CircularDependencyError(f"Cannot sort: cycles detected: {cycles}")
        
        try:
            return list(nx.topological_sort(self.concept_graph.graph))
        except nx.NetworkXError as e:
            raise CircularDependencyError(f"Topological sort failed: {e}")


@dataclass
class SequencingContext:
    """Context for curriculum sequencing."""
    
    user_id: str
    concepts: Dict[str, CurriculumNode]
    user_profile_service: UserProfileService
    dependency_resolver: ConceptDependencyResolver
    strategy: CurriculumStrategy


class CurriculumSequencer:
    """Base class for curriculum sequencing strategies."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence concepts according to strategy.
        
        Args:
            context: Context containing concepts and user info.
        
        Returns:
            Ordered list of curriculum nodes.
        """
        raise NotImplementedError


class BreadthFirstSequencer(CurriculumSequencer):
    """Breadth-first sequencing: cover broad overview first."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence using breadth-first approach.
        
        Concepts with fewer dependencies are prioritized, allowing broad coverage
        before deep dives.
        """
        # Sort by depth (ascending), then by priority (descending)
        sorted_concepts = sorted(
            context.concepts.values(),
            key=lambda c: (c.dependency_depth, -c.priority, c.concept_name)
        )
        
        # Apply topological ordering respecting dependencies
        return self._apply_topological_constraints(context, sorted_concepts)
    
    def _apply_topological_constraints(
        self, context: SequencingContext, concepts: List[CurriculumNode]
    ) -> List[CurriculumNode]:
        """Apply topological constraints while maintaining breadth-first preference."""
        ordered = []
        remaining = set(c.concept_id for c in concepts)
        available = set()
        
        concept_map = {c.concept_id: c for c in concepts}
        
        while remaining:
            # Find newly available concepts (all deps satisfied)
            for concept_id in list(remaining):
                if concept_id not in available:
                    deps = concept_map[concept_id].dependencies
                    if deps.issubset(set(c.concept_id for c in ordered)):
                        available.add(concept_id)
            
            if not available:
                # No available concepts but some remaining: likely circular dependency
                # Add remaining in original order
                for concept_id in remaining:
                    ordered.append(concept_map[concept_id])
                break
            
            # Pick highest priority available concept
            next_concept_id = max(available, key=lambda cid: concept_map[cid].priority)
            ordered.append(concept_map[next_concept_id])
            available.remove(next_concept_id)
            remaining.remove(next_concept_id)
        
        return ordered


class DepthFirstSequencer(CurriculumSequencer):
    """Depth-first sequencing: deep dive into one area."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence using depth-first approach.
        
        Follows dependency chains deeply, completing prerequisite chains
        before moving to alternative topics.
        """
        # Start with concepts that have no dependents (leaf concepts)
        # This ensures we learn prerequisites first
        
        concept_map = {c.concept_id: c for c in context.concepts.values()}
        
        # Find root concepts (no dependencies)
        roots = [c for c in context.concepts.values() if not c.dependencies]
        
        if not roots:
            # No roots found, start with lowest depth
            roots = [min(context.concepts.values(), key=lambda c: c.dependency_depth)]
        
        ordered = []
        visited = set()
        
        for root in sorted(roots, key=lambda c: -c.priority):
            self._dfs_visit(root, ordered, visited, concept_map)
        
        # Add any unvisited concepts
        for concept_id, concept in concept_map.items():
            if concept_id not in visited:
                ordered.append(concept)
                visited.add(concept_id)
        
        return ordered
    
    def _dfs_visit(
        self,
        concept: CurriculumNode,
        ordered: List[CurriculumNode],
        visited: Set[str],
        concept_map: Dict[str, CurriculumNode],
    ):
        """Recursively visit concepts in depth-first order."""
        if concept.concept_id in visited:
            return
        
        visited.add(concept.concept_id)
        
        # Visit all dependents (concepts that depend on this one)
        for dependent_id, dep_concept in concept_map.items():
            if concept.concept_id in dep_concept.dependencies and dependent_id not in visited:
                self._dfs_visit(dep_concept, ordered, visited, concept_map)
        
        ordered.append(concept)


class AdaptiveSequencer(CurriculumSequencer):
    """Adaptive sequencing: dynamic reordering based on learning."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence adaptively based on user mastery levels.
        
        Prioritizes concepts with lower mastery (weak areas) while respecting
        dependencies.
        """
        concept_map = {c.concept_id: c for c in context.concepts.values()}
        
        ordered = []
        remaining = set(c.concept_id for c in context.concepts.values())
        satisfied_deps = set()
        
        while remaining:
            # Find concepts with all dependencies satisfied
            available = []
            for concept_id in remaining:
                concept = concept_map[concept_id]
                if concept.dependencies.issubset(satisfied_deps):
                    available.append(concept)
            
            if not available:
                # No available concepts, use remaining
                for concept_id in remaining:
                    ordered.append(concept_map[concept_id])
                    satisfied_deps.add(concept_id)
                break
            
            # Sort by: lowest mastery, then highest priority, then by name
            next_concept = min(
                available,
                key=lambda c: (c.mastery_level, -c.priority, c.concept_name)
            )
            
            ordered.append(next_concept)
            satisfied_deps.add(next_concept.concept_id)
            remaining.remove(next_concept.concept_id)
        
        return ordered


class SpacedRepetitionSequencer(CurriculumSequencer):
    """Spaced repetition sequencing: strategically timed reviews."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence with spaced repetition insertions.
        
        Interleaves concepts with strategic review points based on
        learning science principles.
        """
        # First get an adaptive base sequence
        base_sequencer = AdaptiveSequencer()
        base_sequence = base_sequencer.sequence(context)
        
        # Insert review points following spaced repetition schedule
        # (1 day, 3 days, 7 days, etc.)
        sequenced = []
        
        for i, concept in enumerate(base_sequence):
            sequenced.append(concept)
            
            # Insert review point after roughly 1/3 and 2/3 of concepts
            if i == len(base_sequence) // 3 or i == (2 * len(base_sequence)) // 3:
                # Add review concepts for previously learned items
                if i > 0:
                    # Could add explicit review nodes here
                    pass
        
        return sequenced


class MasteryBasedSequencer(CurriculumSequencer):
    """Mastery-based sequencing: focus on weak areas."""
    
    def sequence(self, context: SequencingContext) -> List[CurriculumNode]:
        """
        Sequence to focus on mastery of weak areas.
        
        Prioritizes concepts with low mastery while respecting dependencies.
        """
        concept_map = {c.concept_id: c for c in context.concepts.values()}
        
        # Separate concepts by mastery level
        not_started = [c for c in context.concepts.values() if c.mastery_level < 0.2]
        beginner = [c for c in context.concepts.values() if 0.2 <= c.mastery_level < 0.5]
        intermediate = [c for c in context.concepts.values() if 0.5 <= c.mastery_level < 0.8]
        proficient = [c for c in context.concepts.values() if c.mastery_level >= 0.8]
        
        # Arrange in order: not_started, beginner, intermediate, proficient
        # Within each, respect topological constraints
        ordered = []
        for group in [not_started, beginner, intermediate, proficient]:
            group_ids = set(c.concept_id for c in group)
            satisfied_deps = set(c.concept_id for c in ordered)
            
            while group_ids:
                available = [
                    c for c in group
                    if c.concept_id in group_ids and c.dependencies.issubset(satisfied_deps)
                ]
                
                if not available:
                    # Add remaining from this group
                    for concept_id in group_ids:
                        ordered.append(concept_map[concept_id])
                        satisfied_deps.add(concept_id)
                    break
                
                # Pick highest priority available
                next_concept = max(available, key=lambda c: c.priority)
                ordered.append(next_concept)
                satisfied_deps.add(next_concept.concept_id)
                group_ids.remove(next_concept.concept_id)
        
        return ordered


class CurriculumGenerator:
    """
    Main curriculum generator orchestrator.
    
    Coordinates concept graph analysis, dependency resolution, and
    curriculum sequencing using multiple strategies.
    """
    
    SEQUENCERS = {
        CurriculumStrategy.BREADTH_FIRST: BreadthFirstSequencer,
        CurriculumStrategy.DEPTH_FIRST: DepthFirstSequencer,
        CurriculumStrategy.ADAPTIVE: AdaptiveSequencer,
        CurriculumStrategy.SPACED_REPETITION: SpacedRepetitionSequencer,
        CurriculumStrategy.MASTERY_BASED: MasteryBasedSequencer,
    }
    
    def __init__(
        self,
        concept_graph: ConceptGraph,
        user_profile_service: Optional[UserProfileService] = None,
        mastery_estimator: Optional[object] = None,
    ):
        """
        Initialize the curriculum generator.
        
        Args:
            concept_graph: The concept graph to build curricula from.
            user_profile_service: Service for user profile management.
            mastery_estimator: Service for estimating mastery levels.
        """
        self.concept_graph = concept_graph
        self.user_profile_service = user_profile_service
        self.mastery_estimator = mastery_estimator
        self.dependency_resolver = ConceptDependencyResolver(concept_graph)
    
    def build_curriculum_nodes(
        self,
        user_id: Optional[str] = None,
        include_concepts: Optional[List[str]] = None,
        exclude_concepts: Optional[Set[str]] = None,
    ) -> Dict[str, CurriculumNode]:
        """
        Build curriculum nodes from the concept graph.
        
        Args:
            user_id: User ID for personalizing mastery levels.
            include_concepts: List of concept IDs to include (None = all).
            exclude_concepts: Set of concept IDs to exclude.
        
        Returns:
            Dictionary mapping concept IDs to CurriculumNode objects.
        """
        exclude_concepts = exclude_concepts or set()
        curriculum_nodes = {}
        
        for concept_id in self.concept_graph.graph.nodes():
            if concept_id in exclude_concepts:
                continue
            
            if include_concepts and concept_id not in include_concepts:
                continue
            
            concept_node = self.concept_graph.get_concept(concept_id)
            if not concept_node:
                continue
            
            # Get dependencies
            direct_deps = self.dependency_resolver.get_direct_dependencies(concept_id)
            try:
                transitive_deps = self.dependency_resolver.get_transitive_dependencies(concept_id)
            except CircularDependencyError:
                logger.warning(f"Circular dependency detected for concept {concept_id}")
                transitive_deps = set()
            
            depth = self.dependency_resolver.get_dependency_depth(concept_id)
            
            # Get mastery level from user profile if available
            mastery = 0.0
            if user_id and self.user_profile_service:
                profile = self.user_profile_service.get_profile()
                mastery_dict = profile.concept_mastery
                if concept_id in mastery_dict:
                    mastery = mastery_dict[concept_id].mastery_score
            
            # Create curriculum node
            curriculum_node = CurriculumNode(
                concept_id=concept_id,
                concept_name=concept_node.primary_concept.name,
                dependencies=direct_deps,
                priority=float(concept_node.primary_concept.score),
                difficulty=0.5,  # Could be enhanced with better estimation
                mastery_level=mastery,
                dependency_depth=depth,
                prerequisite_ids=list(direct_deps),
                transitive_dependencies=transitive_deps,
            )
            
            curriculum_nodes[concept_id] = curriculum_node
        
        return curriculum_nodes
    
    def generate_learning_path(
        self,
        user_id: str,
        strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE,
        start_concepts: Optional[List[str]] = None,
        target_concepts: Optional[List[str]] = None,
        exclude_concepts: Optional[Set[str]] = None,
    ) -> LearningPath:
        """
        Generate a personalized learning path for a user.
        
        Args:
            user_id: The user ID.
            strategy: The sequencing strategy to use.
            start_concepts: Concepts to start from (if partial path).
            target_concepts: Concepts to target as end goal.
            exclude_concepts: Concepts to exclude from path.
        
        Returns:
            A LearningPath object representing the sequenced curriculum.
        
        Raises:
            CircularDependencyError: If circular dependencies prevent path generation.
        """
        # Check for cycles
        if self.dependency_resolver.has_cycles():
            cycles = self.dependency_resolver.detect_cycles()
            logger.warning(f"Cycles detected in concept graph: {cycles}")
        
        # Build curriculum nodes
        include_concepts = target_concepts
        curriculum_nodes = self.build_curriculum_nodes(
            user_id=user_id,
            include_concepts=include_concepts,
            exclude_concepts=exclude_concepts,
        )
        
        if not curriculum_nodes:
            raise ValueError("No concepts available for curriculum generation")
        
        # Determine start point for partial paths
        if start_concepts:
            # Remove start concepts' prerequisites from the path
            start_ids = set(start_concepts)
            for concept_id in start_ids:
                if concept_id in curriculum_nodes:
                    curriculum_nodes[concept_id].mastery_level = 1.0
        
        # Create sequencing context
        sequencer_class = self.SEQUENCERS.get(strategy, AdaptiveSequencer)
        sequencer = sequencer_class()
        
        context = SequencingContext(
            user_id=user_id,
            concepts=curriculum_nodes,
            user_profile_service=self.user_profile_service,
            dependency_resolver=self.dependency_resolver,
            strategy=strategy,
        )
        
        # Generate sequence
        sequenced_concepts = sequencer.sequence(context)
        
        # Calculate total estimated time
        total_time = sum(c.estimated_time_minutes for c in sequenced_concepts)
        
        # Create learning path
        path = LearningPath(
            path_id=f"path_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            concepts=sequenced_concepts,
            strategy=strategy,
            estimated_total_time_minutes=total_time,
            metadata={
                "start_concepts": start_concepts or [],
                "target_concepts": target_concepts or [],
                "excluded_concepts": list(exclude_concepts or []),
            },
        )
        
        return path
    
    def update_learning_path_progress(
        self,
        path: LearningPath,
        concept_id: str,
        mastery_level: float,
        time_spent: float,
    ) -> None:
        """
        Update a learning path with user progress.
        
        Args:
            path: The learning path to update.
            concept_id: The concept that was completed.
            mastery_level: The mastery level achieved (0-1).
            time_spent: Time spent on this concept (minutes).
        """
        path.completed_concepts.add(concept_id)
        path.actual_time_spent_minutes += time_spent
        path.last_updated_at = datetime.now()
        
        # Update progress percentage
        total_concepts = len(path.concepts)
        if total_concepts > 0:
            path.progress = len(path.completed_concepts) / total_concepts
        
        # Check if path is complete
        if path.progress >= 1.0:
            path.status = PathProgressState.COMPLETED
            path.completed_at = datetime.now()
        else:
            if path.status == PathProgressState.NOT_STARTED:
                path.status = PathProgressState.IN_PROGRESS
                path.started_at = datetime.now()
    
    def estimate_time_to_completion(
        self,
        path: LearningPath,
        current_mastery_levels: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate time to completion for a learning path.
        
        Args:
            path: The learning path.
            current_mastery_levels: Current mastery levels (optional).
        
        Returns:
            Tuple of (estimated_time_minutes, confidence_0_to_1).
        """
        remaining_time = sum(
            c.estimated_time_minutes
            for c in path.concepts
            if c.concept_id not in path.completed_concepts
        )
        
        # Adjust based on average learning speed
        avg_actual_time = path.actual_time_spent_minutes / len(path.completed_concepts) \
            if path.completed_concepts else 0
        
        learning_rate = (avg_actual_time / sum(
            path.concepts[i].estimated_time_minutes
            for i in range(len(path.completed_concepts))
        )) if path.completed_concepts else 1.0
        
        adjusted_time = remaining_time * max(0.5, min(2.0, learning_rate))
        
        # Confidence based on completion percentage
        confidence = min(1.0, path.progress)
        
        return (adjusted_time, confidence)
    
    def suggest_strategy(self, user_id: str) -> CurriculumStrategy:
        """
        Suggest optimal curriculum strategy for a user.
        
        Args:
            user_id: The user ID.
        
        Returns:
            Recommended CurriculumStrategy.
        """
        if not self.user_profile_service:
            return CurriculumStrategy.ADAPTIVE
        
        profile = self.user_profile_service.get_profile()
        
        # Check learning pace preference
        if hasattr(profile, 'learning_preferences'):
            prefs = profile.learning_preferences
            if hasattr(prefs, 'learning_pace'):
                pace = prefs.learning_pace.value
                if pace == "fast":
                    return CurriculumStrategy.BREADTH_FIRST
                elif pace == "slow":
                    return CurriculumStrategy.DEPTH_FIRST
        
        # Check mastery distribution
        mastery_scores = list(profile.concept_mastery.values())
        if mastery_scores:
            avg_mastery = sum(m.mastery_score for m in mastery_scores) / len(mastery_scores)
            if avg_mastery < 0.3:
                return CurriculumStrategy.MASTERY_BASED
        
        return CurriculumStrategy.ADAPTIVE
