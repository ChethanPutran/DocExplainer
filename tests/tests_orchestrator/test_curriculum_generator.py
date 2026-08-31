"""Tests for the curriculum generator module."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Set

from src.core.orchestrator.curriculum_generator import (
    CurriculumGenerator,
    ConceptDependencyResolver,
    CircularDependencyError,
    BreadthFirstSequencer,
    DepthFirstSequencer,
    AdaptiveSequencer,
    MasteryBasedSequencer,
    SequencingContext,
)
from src.core.orchestrator.models import (
    CurriculumStrategy,
    CurriculumNode,
    LearningPath,
    PathProgressState,
)
from src.core.knowledge.models.concept import Concept
from src.core.knowledge.models.relationship import ConceptNode
from src.core.knowledge.models.graph import ConceptGraph
from src.core.user.models.user_profile import UserProfile
from src.core.user.services.user_profile_service import UserProfileService


class TestConceptDependencyResolver:
    """Test concept dependency resolution."""
    
    @pytest.fixture
    def concept_graph_simple(self) -> ConceptGraph:
        """Create a simple concept graph for testing."""
        graph = ConceptGraph()
        
        # Create concepts: A -> B -> C (A depends on B, B depends on C)
        concepts = {
            "A": Concept(name="A", score=1.0),
            "B": Concept(name="B", score=0.8),
            "C": Concept(name="C", score=0.6),
        }
        
        # Add nodes
        for cid, concept in concepts.items():
            node = ConceptNode(primary_concept=concept)
            graph.add_concept_node(node)
        
        # Add edges: C -> B -> A (C is prerequisite of B, B is prerequisite of A)
        from src.core.knowledge.models.relationship import ConceptRelationship, ConceptNodeRelationship
        
        rel_cb = ConceptRelationship(
            concept1=concepts["C"],
            concept2=concepts["B"],
            relation="prerequisite",
            strength=1.0
        )
        rel_ba = ConceptRelationship(
            concept1=concepts["B"],
            concept2=concepts["A"],
            relation="prerequisite",
            strength=1.0
        )
        
        edge_cb = ConceptNodeRelationship(
            concept1=graph.get_concept("C"),
            concept2=graph.get_concept("B"),
            relationship=rel_cb
        )
        edge_ba = ConceptNodeRelationship(
            concept1=graph.get_concept("B"),
            concept2=graph.get_concept("A"),
            relationship=rel_ba
        )
        
        graph.add_relationship(graph.get_concept("C"), graph.get_concept("B"), edge_cb)
        graph.add_relationship(graph.get_concept("B"), graph.get_concept("A"), edge_ba)
        
        return graph
    
    def test_get_direct_dependencies(self, concept_graph_simple):
        """Test getting direct dependencies."""
        resolver = ConceptDependencyResolver(concept_graph_simple)
        
        deps_a = resolver.get_direct_dependencies("A")
        assert "B" in deps_a, "A should depend on B"
        
        deps_b = resolver.get_direct_dependencies("B")
        assert "C" in deps_b, "B should depend on C"
        
        deps_c = resolver.get_direct_dependencies("C")
        assert len(deps_c) == 0, "C should have no dependencies"
    
    def test_get_transitive_dependencies(self, concept_graph_simple):
        """Test getting transitive dependencies."""
        resolver = ConceptDependencyResolver(concept_graph_simple)
        
        deps_a = resolver.get_transitive_dependencies("A")
        assert "B" in deps_a and "C" in deps_a, "A should transitively depend on B and C"
        
        deps_b = resolver.get_transitive_dependencies("B")
        assert "C" in deps_b, "B should transitively depend on C"
        
        deps_c = resolver.get_transitive_dependencies("C")
        assert len(deps_c) == 0, "C should have no transitive dependencies"
    
    def test_get_dependency_depth(self, concept_graph_simple):
        """Test getting dependency depth."""
        resolver = ConceptDependencyResolver(concept_graph_simple)
        
        assert resolver.get_dependency_depth("C") == 0, "C has depth 0"
        assert resolver.get_dependency_depth("B") == 1, "B has depth 1"
        assert resolver.get_dependency_depth("A") == 2, "A has depth 2"
    
    def test_topological_sort(self, concept_graph_simple):
        """Test topological sorting."""
        resolver = ConceptDependencyResolver(concept_graph_simple)
        
        sorted_concepts = resolver.topological_sort()
        
        assert len(sorted_concepts) == 3
        # C should come before B, B should come before A
        assert sorted_concepts.index("C") < sorted_concepts.index("B")
        assert sorted_concepts.index("B") < sorted_concepts.index("A")
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        graph = ConceptGraph()
        
        # Create circular dependency: A -> B -> C -> A
        from src.core.knowledge.models.relationship import ConceptRelationship, ConceptNodeRelationship
        
        concepts = {
            "A": Concept(name="A"),
            "B": Concept(name="B"),
            "C": Concept(name="C"),
        }
        
        for cid, concept in concepts.items():
            node = ConceptNode(primary_concept=concept)
            graph.add_concept_node(node)
        
        # Create circular edges
        edges_data = [
            ("A", "B"), ("B", "C"), ("C", "A")
        ]
        
        for source, target in edges_data:
            rel = ConceptRelationship(concepts[source], concepts[target])
            edge = ConceptNodeRelationship(
                concept1=graph.get_concept(source),
                concept2=graph.get_concept(target),
                relationship=rel
            )
            graph.add_relationship(graph.get_concept(source), graph.get_concept(target), edge)
        
        resolver = ConceptDependencyResolver(graph)
        
        # Should detect cycles
        assert resolver.has_cycles()
        cycles = resolver.detect_cycles()
        assert len(cycles) > 0


class TestCurriculumNode:
    """Test curriculum node models."""
    
    def test_curriculum_node_creation(self):
        """Test creating a curriculum node."""
        node = CurriculumNode(
            concept_id="python-basics",
            concept_name="Python Basics",
            dependencies={"variables"},
            estimated_time_minutes=30.0,
            priority=0.9,
            difficulty=0.3,
            mastery_level=0.5,
        )
        
        assert node.concept_id == "python-basics"
        assert node.concept_name == "Python Basics"
        assert "variables" in node.dependencies
        assert node.priority == 0.9
    
    def test_curriculum_node_serialization(self):
        """Test serializing and deserializing curriculum nodes."""
        original = CurriculumNode(
            concept_id="test",
            concept_name="Test Concept",
            dependencies={"dep1", "dep2"},
            priority=0.8,
            mastery_level=0.6,
        )
        
        # Serialize
        data = original.to_dict()
        assert data["concept_id"] == "test"
        
        # Deserialize
        restored = CurriculumNode.from_dict(data)
        assert restored.concept_id == original.concept_id
        assert restored.concept_name == original.concept_name
        assert restored.priority == original.priority


class TestLearningPath:
    """Test learning path models."""
    
    def test_learning_path_creation(self):
        """Test creating a learning path."""
        concepts = [
            CurriculumNode("concept1", "Concept 1"),
            CurriculumNode("concept2", "Concept 2"),
        ]
        
        path = LearningPath(
            path_id="path1",
            user_id="user1",
            concepts=concepts,
            strategy=CurriculumStrategy.BREADTH_FIRST,
            estimated_total_time_minutes=60.0,
        )
        
        assert path.path_id == "path1"
        assert len(path.concepts) == 2
        assert path.progress == 0.0
    
    def test_learning_path_progress(self):
        """Test updating learning path progress."""
        concepts = [
            CurriculumNode("c1", "Concept 1"),
            CurriculumNode("c2", "Concept 2"),
        ]
        
        path = LearningPath(
            path_id="path1",
            user_id="user1",
            concepts=concepts,
            estimated_total_time_minutes=20.0,
        )
        
        path.completed_concepts.add("c1")
        path.progress = 0.5
        
        assert "c1" in path.completed_concepts
        assert path.progress == 0.5
    
    def test_learning_path_serialization(self):
        """Test serializing and deserializing learning paths."""
        concepts = [CurriculumNode("c1", "Concept 1")]
        original = LearningPath(
            path_id="path1",
            user_id="user1",
            concepts=concepts,
        )
        
        data = original.to_dict()
        restored = LearningPath.from_dict(data)
        
        assert restored.path_id == original.path_id
        assert restored.user_id == original.user_id


class TestBreadthFirstSequencer:
    """Test breadth-first sequencing strategy."""
    
    def test_breadth_first_basic(self):
        """Test breadth-first sequencing."""
        # Create concepts with different depths
        concepts = {
            "c1": CurriculumNode("c1", "C1", dependencies=set(), dependency_depth=0),
            "c2": CurriculumNode("c2", "C2", dependencies={"c1"}, dependency_depth=1),
            "c3": CurriculumNode("c3", "C3", dependencies={"c1"}, dependency_depth=1),
            "c4": CurriculumNode("c4", "C4", dependencies={"c2", "c3"}, dependency_depth=2),
        }
        
        sequencer = BreadthFirstSequencer()
        sequenced = sequencer.sequence(
            SequencingContext(
                user_id="user1",
                concepts=concepts,
                user_profile_service=None,
                dependency_resolver=None,
                strategy=CurriculumStrategy.BREADTH_FIRST,
            )
        )
        
        # c1 should come first (depth 0)
        assert sequenced[0].concept_id == "c1"
        
        # c2 and c3 should come before c4
        assert any(c.concept_id == "c2" for c in sequenced[:3])
        assert any(c.concept_id == "c3" for c in sequenced[:3])


class TestDepthFirstSequencer:
    """Test depth-first sequencing strategy."""
    
    def test_depth_first_basic(self):
        """Test depth-first sequencing."""
        concepts = {
            "c1": CurriculumNode("c1", "C1", dependencies=set()),
            "c2": CurriculumNode("c2", "C2", dependencies={"c1"}),
            "c3": CurriculumNode("c3", "C3", dependencies={"c1"}),
        }
        
        sequencer = DepthFirstSequencer()
        sequenced = sequencer.sequence(
            SequencingContext(
                user_id="user1",
                concepts=concepts,
                user_profile_service=None,
                dependency_resolver=None,
                strategy=CurriculumStrategy.DEPTH_FIRST,
            )
        )
        
        assert len(sequenced) == 3
        # All should be present
        concept_ids = {c.concept_id for c in sequenced}
        assert concept_ids == {"c1", "c2", "c3"}


class TestAdaptiveSequencer:
    """Test adaptive sequencing strategy."""
    
    def test_adaptive_prioritizes_low_mastery(self):
        """Test that adaptive sequencer prioritizes low mastery concepts."""
        concepts = {
            "c1": CurriculumNode("c1", "C1", dependencies=set(), mastery_level=0.0),
            "c2": CurriculumNode("c2", "C2", dependencies=set(), mastery_level=0.8),
            "c3": CurriculumNode("c3", "C3", dependencies=set(), mastery_level=0.5),
        }
        
        sequencer = AdaptiveSequencer()
        sequenced = sequencer.sequence(
            SequencingContext(
                user_id="user1",
                concepts=concepts,
                user_profile_service=None,
                dependency_resolver=None,
                strategy=CurriculumStrategy.ADAPTIVE,
            )
        )
        
        # c1 (mastery 0.0) should come first
        assert sequenced[0].concept_id == "c1"
        # c2 (mastery 0.8) should come last
        assert sequenced[-1].concept_id == "c2"


class TestCurriculumGenerator:
    """Test the main curriculum generator."""
    
    @pytest.fixture
    def setup_generator(self) -> CurriculumGenerator:
        """Set up a curriculum generator with sample data."""
        graph = ConceptGraph()
        
        from src.core.knowledge.models.relationship import ConceptNode as CN
        
        concepts = {
            "python": Concept(name="Python", score=0.9),
            "variables": Concept(name="Variables", score=0.8),
            "loops": Concept(name="Loops", score=0.7),
        }
        
        for cid, concept in concepts.items():
            node = CN(primary_concept=concept)
            graph.add_concept_node(node)
        
        return CurriculumGenerator(graph)
    
    def test_build_curriculum_nodes(self, setup_generator):
        """Test building curriculum nodes."""
        nodes = setup_generator.build_curriculum_nodes()
        
        assert len(nodes) == 3
        assert "python" in nodes
        assert nodes["python"].concept_name == "Python"
    
    def test_generate_learning_path_basic(self, setup_generator):
        """Test generating a basic learning path."""
        path = setup_generator.generate_learning_path(
            user_id="user1",
            strategy=CurriculumStrategy.ADAPTIVE,
        )
        
        assert path.user_id == "user1"
        assert path.strategy == CurriculumStrategy.ADAPTIVE
        assert len(path.concepts) > 0
        assert path.status == PathProgressState.NOT_STARTED
    
    def test_estimate_time_to_completion(self, setup_generator):
        """Test time to completion estimation."""
        path = setup_generator.generate_learning_path("user1")
        
        estimated_time, confidence = setup_generator.estimate_time_to_completion(path)
        
        assert estimated_time > 0
        assert 0 <= confidence <= 1
        assert estimated_time == path.estimated_total_time_minutes
    
    def test_update_path_progress(self, setup_generator):
        """Test updating learning path progress."""
        path = setup_generator.generate_learning_path("user1")
        
        if len(path.concepts) > 0:
            setup_generator.update_learning_path_progress(
                path,
                path.concepts[0].concept_id,
                mastery_level=0.8,
                time_spent=15.0,
            )
            
            assert path.concepts[0].concept_id in path.completed_concepts
            assert path.actual_time_spent_minutes == 15.0
            assert path.status == PathProgressState.IN_PROGRESS


class TestCurriculumStrategy:
    """Test curriculum strategy enum."""
    
    def test_strategy_values(self):
        """Test that all strategies have unique values."""
        strategies = [
            CurriculumStrategy.BREADTH_FIRST,
            CurriculumStrategy.DEPTH_FIRST,
            CurriculumStrategy.ADAPTIVE,
            CurriculumStrategy.SPACED_REPETITION,
            CurriculumStrategy.MASTERY_BASED,
        ]
        
        values = [s.value for s in strategies]
        assert len(values) == len(set(values)), "All strategies should have unique values"


class TestPathProgressState:
    """Test path progress state enum."""
    
    def test_progress_state_values(self):
        """Test that all progress states are valid."""
        states = [
            PathProgressState.NOT_STARTED,
            PathProgressState.IN_PROGRESS,
            PathProgressState.COMPLETED,
            PathProgressState.PAUSED,
            PathProgressState.ABANDONED,
        ]
        
        assert len(states) == 5


def test_learning_path_serialization_roundtrip():
    """Test full serialization roundtrip."""
    concepts = [
        CurriculumNode("c1", "Concept 1", priority=0.9),
        CurriculumNode("c2", "Concept 2", priority=0.7),
    ]
    
    original = LearningPath(
        path_id="test-path",
        user_id="test-user",
        concepts=concepts,
        strategy=CurriculumStrategy.DEPTH_FIRST,
        progress=0.5,
        completed_concepts={"c1"},
    )
    
    # Serialize to dict
    data = original.to_dict()
    
    # Deserialize from dict
    restored = LearningPath.from_dict(data)
    
    # Verify key fields
    assert restored.path_id == original.path_id
    assert restored.user_id == original.user_id
    assert restored.strategy == original.strategy
    assert restored.progress == original.progress
    assert restored.completed_concepts == original.completed_concepts
