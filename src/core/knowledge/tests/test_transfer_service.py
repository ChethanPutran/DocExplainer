"""Tests for multi-document knowledge transfer."""

import pytest
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import transfer service models directly to avoid circular dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("transfer_models", 
    Path(__file__).parent.parent.parent / "knowledge" / "models" / "transfer_models.py")
transfer_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_models)

TransferConfig = transfer_models.TransferConfig
ConceptAlignmentType = transfer_models.ConceptAlignmentType
ConceptMapping = transfer_models.ConceptMapping
DocumentTransfer = transfer_models.DocumentTransfer
TransferAnalysisResult = transfer_models.TransferAnalysisResult


class TestConceptMapping:
    """Test ConceptMapping model."""

    def test_concept_mapping_creation(self):
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

    def test_concept_mapping_to_dict(self):
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


class TestDocumentTransfer:
    """Test DocumentTransfer model."""

    def test_document_transfer_creation(self):
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

    def test_document_transfer_statistics(self):
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

    def test_document_transfer_to_dict(self):
        """Test converting transfer to dict."""
        mappings = [
            ConceptMapping(
                source_concept="c1",
                target_concept="c2",
                source_doc="d1",
                target_doc="d2",
                similarity_score=0.8,
                transfer_score=0.75,
            ),
        ]
        
        transfer = DocumentTransfer(
            source_doc="d1",
            target_doc="d2",
            concept_mappings=mappings,
            overall_score=0.75,
        )
        
        transfer_dict = transfer.to_dict()
        
        assert isinstance(transfer_dict, dict)
        assert transfer_dict["source_doc"] == "d1"
        assert len(transfer_dict["concept_mappings"]) == 1


class TestTransferConfig:
    """Test transfer configuration model."""

    def test_config_defaults(self):
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

    def test_config_custom_values(self):
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

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TransferConfig(similarity_threshold=0.75)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["similarity_threshold"] == 0.75
        assert "discount_factor" in config_dict
        assert "embedding_model" in config_dict

    def test_config_embedding_model(self):
        """Test embedding model configuration."""
        custom_model = "sentence-transformers/mpnet-base-v2"
        config = TransferConfig(embedding_model=custom_model)
        
        assert config.embedding_model == custom_model


class TestTransferAnalysisResult:
    """Test TransferAnalysisResult model."""

    def test_transfer_analysis_result_creation(self):
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

    def test_transfer_analysis_result_to_dict(self):
        """Test converting result to dict."""
        transfers = [
            DocumentTransfer(
                source_doc="d1",
                target_doc="d2",
                concept_mappings=[],
                overall_score=0.7,
            ),
        ]
        
        result = TransferAnalysisResult(
            transfers=transfers,
            total_documents=2,
            total_mappings=3,
            average_transfer_score=0.7,
            computation_time_ms=100.0,
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["total_documents"] == 2
        assert result_dict["total_mappings"] == 3
        assert len(result_dict["transfers"]) == 1


class TestConceptAlignmentType:
    """Test ConceptAlignmentType enum."""

    def test_alignment_types(self):
        """Test alignment type enum values."""
        assert ConceptAlignmentType.IDENTICAL.value == "identical"
        assert ConceptAlignmentType.EQUIVALENT.value == "equivalent"
        assert ConceptAlignmentType.SIMILAR.value == "similar"
        assert ConceptAlignmentType.HIERARCHICAL.value == "hierarchical"
        assert ConceptAlignmentType.ALIAS.value == "alias"

    def test_alignment_type_comparison(self):
        """Test alignment type comparisons."""
        t1 = ConceptAlignmentType.IDENTICAL
        t2 = ConceptAlignmentType.IDENTICAL
        t3 = ConceptAlignmentType.EQUIVALENT
        
        assert t1 == t2
        assert t1 != t3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
