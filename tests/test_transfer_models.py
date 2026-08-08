"""Tests for multi-document knowledge transfer - Standalone version."""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import transfer service models directly to avoid circular dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("transfer_models", 
    project_root / "src" / "core" / "knowledge" / "models" / "transfer_models.py")
transfer_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_models)

TransferConfig = transfer_models.TransferConfig
ConceptAlignmentType = transfer_models.ConceptAlignmentType
ConceptMapping = transfer_models.ConceptMapping
DocumentTransfer = transfer_models.DocumentTransfer
TransferAnalysisResult = transfer_models.TransferAnalysisResult


def test_concept_mapping_creation():
    """Test creating a concept mapping."""
    mapping = ConceptMapping(
        source_concept="machine learning",
        target_concept="deep learning",
        source_doc="doc1",
        target_doc="doc2",
        similarity_score=0.85,
        transfer_score=0.8,
    )
    
    assert mapping.source_concept == "machine learning"
    assert mapping.target_concept == "deep learning"
    assert mapping.transfer_score == 0.8
    assert mapping.alignment_type == ConceptAlignmentType.EQUIVALENT
    print("✓ test_concept_mapping_creation passed")


def test_concept_mapping_to_dict():
    """Test converting mapping to dict."""
    mapping = ConceptMapping(
        source_concept="A",
        target_concept="B",
        source_doc="d1",
        target_doc="d2",
        similarity_score=0.9,
        transfer_score=0.85,
        confidence=0.95,
    )
    
    mapping_dict = mapping.to_dict()
    
    assert isinstance(mapping_dict, dict)
    assert mapping_dict["source_concept"] == "A"
    assert mapping_dict["confidence"] == 0.95
    assert "transfer_score" in mapping_dict
    print("✓ test_concept_mapping_to_dict passed")


def test_document_transfer_creation():
    """Test creating document transfer."""
    mappings = [
        ConceptMapping(
            source_concept="c1",
            target_concept="c2",
            source_doc="d1",
            target_doc="d2",
            similarity_score=0.8,
            transfer_score=0.75,
        ),
        ConceptMapping(
            source_concept="c3",
            target_concept="c4",
            source_doc="d1",
            target_doc="d2",
            similarity_score=0.7,
            transfer_score=0.65,
        ),
    ]
    
    transfer = DocumentTransfer(
        source_doc="d1",
        target_doc="d2",
        concept_mappings=mappings,
        overall_score=0.7,
    )
    
    assert transfer.total_mappings == 2
    assert transfer.source_doc == "d1"
    assert transfer.overall_score == 0.7
    print("✓ test_document_transfer_creation passed")


def test_document_transfer_statistics():
    """Test document transfer statistics calculation."""
    mappings = [
        ConceptMapping(
            source_concept="c1",
            target_concept="c2",
            source_doc="d1",
            target_doc="d2",
            similarity_score=0.8,
            transfer_score=0.75,
            confidence=0.9,
        ),
        ConceptMapping(
            source_concept="c3",
            target_concept="c4",
            source_doc="d1",
            target_doc="d2",
            similarity_score=0.5,
            transfer_score=0.45,
            confidence=0.6,
        ),
    ]
    
    transfer = DocumentTransfer(
        source_doc="d1",
        target_doc="d2",
        concept_mappings=mappings,
        overall_score=0.6,
    )
    
    # Should calculate high confidence mappings
    assert transfer.high_confidence_mappings == 1
    assert transfer.total_mappings == 2
    print("✓ test_document_transfer_statistics passed")


def test_transfer_config_defaults():
    """Test configuration defaults."""
    config = TransferConfig()
    
    assert config.similarity_threshold == 0.6
    assert config.discount_factor == 0.8
    assert config.min_confidence == 0.5
    assert config.use_historical_data is True
    assert config.max_mappings_per_pair == 100
    assert config.enable_hierarchical_detection is True
    assert config.enable_alias_detection is True
    assert config.cache_embeddings is True
    print("✓ test_transfer_config_defaults passed")


def test_transfer_config_custom():
    """Test configuration with custom values."""
    config = TransferConfig(
        similarity_threshold=0.7,
        discount_factor=0.9,
        min_confidence=0.6,
        max_mappings_per_pair=50,
    )
    
    assert config.similarity_threshold == 0.7
    assert config.discount_factor == 0.9
    assert config.min_confidence == 0.6
    assert config.max_mappings_per_pair == 50
    print("✓ test_transfer_config_custom passed")


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = TransferConfig(similarity_threshold=0.75)
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["similarity_threshold"] == 0.75
    assert "discount_factor" in config_dict
    assert "embedding_model" in config_dict
    print("✓ test_config_to_dict passed")


def test_transfer_analysis_result_creation():
    """Test creating analysis result."""
    transfers = [
        DocumentTransfer(
            source_doc="d1",
            target_doc="d2",
            concept_mappings=[],
            overall_score=0.7,
        ),
        DocumentTransfer(
            source_doc="d1",
            target_doc="d3",
            concept_mappings=[],
            overall_score=0.5,
        ),
    ]
    
    result = TransferAnalysisResult(
        transfers=transfers,
        total_documents=3,
        total_mappings=5,
        average_transfer_score=0.6,
        computation_time_ms=150.5,
    )
    
    assert result.total_documents == 3
    assert result.total_mappings == 5
    assert result.average_transfer_score == 0.6
    assert result.computation_time_ms == 150.5
    assert len(result.transfers) == 2
    print("✓ test_transfer_analysis_result_creation passed")


def test_alignment_types():
    """Test alignment type enum values."""
    assert ConceptAlignmentType.IDENTICAL.value == "identical"
    assert ConceptAlignmentType.EQUIVALENT.value == "equivalent"
    assert ConceptAlignmentType.SIMILAR.value == "similar"
    assert ConceptAlignmentType.HIERARCHICAL.value == "hierarchical"
    assert ConceptAlignmentType.ALIAS.value == "alias"
    print("✓ test_alignment_types passed")


def test_alignment_type_comparison():
    """Test alignment type comparisons."""
    t1 = ConceptAlignmentType.IDENTICAL
    t2 = ConceptAlignmentType.IDENTICAL
    t3 = ConceptAlignmentType.EQUIVALENT
    
    assert t1 == t2
    assert t1 != t3
    print("✓ test_alignment_type_comparison passed")


if __name__ == "__main__":
    tests = [
        test_concept_mapping_creation,
        test_concept_mapping_to_dict,
        test_document_transfer_creation,
        test_document_transfer_statistics,
        test_transfer_config_defaults,
        test_transfer_config_custom,
        test_config_to_dict,
        test_transfer_analysis_result_creation,
        test_alignment_types,
        test_alignment_type_comparison,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests completed: {len(tests) - failed}/{len(tests)} passed")
    if failed == 0:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print(f"{failed} tests failed")
        sys.exit(1)
