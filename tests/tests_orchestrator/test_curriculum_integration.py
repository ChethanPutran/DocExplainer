"""
Simple integration test for curriculum generator.
Tests core functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_curriculum_models():
    """Test curriculum models can be instantiated."""
    from src.core.orchestrator.models.curriculum_models import (
        CurriculumStrategy,
        CurriculumNode,
        LearningPath,
        PathProgressState,
    )
    
    # Test strategy enum
    assert CurriculumStrategy.BREADTH_FIRST.value == "breadth_first"
    assert CurriculumStrategy.DEPTH_FIRST.value == "depth_first"
    assert CurriculumStrategy.ADAPTIVE.value == "adaptive"
    assert CurriculumStrategy.SPACED_REPETITION.value == "spaced_repetition"
    assert CurriculumStrategy.MASTERY_BASED.value == "mastery_based"
    print("✓ CurriculumStrategy enum works")
    
    # Test CurriculumNode
    node = CurriculumNode(
        concept_id="test-concept",
        concept_name="Test Concept",
        dependencies={"dep1", "dep2"},
        estimated_time_minutes=30.0,
        priority=0.8,
        difficulty=0.5,
        mastery_level=0.6,
    )
    assert node.concept_id == "test-concept"
    assert node.priority == 0.8
    assert "dep1" in node.dependencies
    print("✓ CurriculumNode creation works")
    
    # Test serialization
    data = node.to_dict()
    restored = CurriculumNode.from_dict(data)
    assert restored.concept_id == node.concept_id
    assert restored.priority == node.priority
    print("✓ CurriculumNode serialization works")
    
    # Test LearningPath
    path = LearningPath(
        path_id="test-path",
        user_id="test-user",
        concepts=[node],
        strategy=CurriculumStrategy.ADAPTIVE,
    )
    assert path.path_id == "test-path"
    assert len(path.concepts) == 1
    print("✓ LearningPath creation works")
    
    # Test LearningPath methods
    current = path.get_current_concept()
    assert current is not None
    assert current.concept_id == "test-concept"
    print("✓ LearningPath methods work")
    
    # Test LearningPath serialization
    path_data = path.to_dict()
    path_restored = LearningPath.from_dict(path_data)
    assert path_restored.path_id == path.path_id
    assert path_restored.user_id == path.user_id
    print("✓ LearningPath serialization works")
    
    # Test PathProgressState
    assert PathProgressState.NOT_STARTED.value == "not_started"
    assert PathProgressState.IN_PROGRESS.value == "in_progress"
    assert PathProgressState.COMPLETED.value == "completed"
    print("✓ PathProgressState enum works")


def test_curriculum_sequencers_basic():
    """Test that sequencer classes can be instantiated."""
    from src.core.orchestrator.curriculum_generator import (
        BreadthFirstSequencer,
        DepthFirstSequencer,
        AdaptiveSequencer,
        SpacedRepetitionSequencer,
        MasteryBasedSequencer,
    )
    
    sequencers = [
        BreadthFirstSequencer(),
        DepthFirstSequencer(),
        AdaptiveSequencer(),
        SpacedRepetitionSequencer(),
        MasteryBasedSequencer(),
    ]
    
    for sequencer in sequencers:
        assert hasattr(sequencer, 'sequence')
        print(f"✓ {sequencer.__class__.__name__} can be instantiated")


def test_dependency_resolver_basic():
    """Test ConceptDependencyResolver can be instantiated."""
    from src.core.orchestrator.curriculum_generator import ConceptDependencyResolver
    from unittest.mock import Mock
    
    # Create a mock graph
    mock_graph = Mock()
    mock_graph.graph = Mock()
    mock_graph.graph.predecessors = Mock(return_value=[])
    mock_graph.has_concept = Mock(return_value=True)
    mock_graph.get_concept = Mock(return_value=None)
    
    resolver = ConceptDependencyResolver(mock_graph)
    assert resolver is not None
    print("✓ ConceptDependencyResolver instantiation works")


def test_curriculum_generator_structure():
    """Test CurriculumGenerator class structure."""
    from src.core.orchestrator.curriculum_generator import CurriculumGenerator
    
    # Check that the class has required attributes and methods
    required_methods = [
        'build_curriculum_nodes',
        'generate_learning_path',
        'update_learning_path_progress',
        'estimate_time_to_completion',
        'suggest_strategy',
    ]
    
    for method in required_methods:
        assert hasattr(CurriculumGenerator, method), f"Missing method: {method}"
        print(f"✓ CurriculumGenerator.{method} exists")
    
    # Check SEQUENCERS mapping
    assert hasattr(CurriculumGenerator, 'SEQUENCERS')
    assert len(CurriculumGenerator.SEQUENCERS) == 5
    print("✓ CurriculumGenerator.SEQUENCERS has all strategies")


def test_imports():
    """Test that main module imports work."""
    try:
        from src.core.orchestrator import (
            CurriculumGenerator,
            ConceptDependencyResolver,
            CircularDependencyError,
            CurriculumStrategy,
            CurriculumNode,
            LearningPath,
            PathProgressState,
        )
        print("✓ All main imports work")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        raise


if __name__ == "__main__":
    print("Running curriculum generator integration tests...\n")
    
    try:
        test_curriculum_models()
        print()
        test_curriculum_sequencers_basic()
        print()
        test_dependency_resolver_basic()
        print()
        test_curriculum_generator_structure()
        print()
        test_imports()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
