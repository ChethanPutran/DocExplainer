"""Models for multi-document knowledge transfer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ConceptAlignmentType(str, Enum):
    """Types of concept alignments."""
    IDENTICAL = "identical"
    EQUIVALENT = "equivalent"
    SIMILAR = "similar"
    HIERARCHICAL = "hierarchical"
    ALIAS = "alias"


@dataclass
class ConceptMapping:
    """Represents a mapping between two concepts across documents."""
    source_concept: str
    target_concept: str
    source_doc: str
    target_doc: str
    similarity_score: float  # 0-1, based on semantic similarity
    transfer_score: float  # 0-1, effectiveness of transfer
    alignment_type: ConceptAlignmentType = ConceptAlignmentType.EQUIVALENT
    confidence: float = 0.5  # Confidence in this mapping
    synonyms: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_concept": self.source_concept,
            "target_concept": self.target_concept,
            "source_doc": self.source_doc,
            "target_doc": self.target_doc,
            "similarity_score": self.similarity_score,
            "transfer_score": self.transfer_score,
            "alignment_type": self.alignment_type.value,
            "confidence": self.confidence,
            "synonyms": self.synonyms,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }


@dataclass
class DocumentTransfer:
    """Represents transfer analysis between two documents."""
    source_doc: str
    target_doc: str
    concept_mappings: List[ConceptMapping]
    overall_score: float  # 0-1, average transfer effectiveness
    total_mappings: int = 0
    high_confidence_mappings: int = 0
    domain_similarity: float = 0.5  # How similar are the domains
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total_mappings = len(self.concept_mappings)
        self.high_confidence_mappings = sum(
            1 for m in self.concept_mappings if m.confidence > 0.7
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_doc": self.source_doc,
            "target_doc": self.target_doc,
            "concept_mappings": [m.to_dict() for m in self.concept_mappings],
            "overall_score": self.overall_score,
            "total_mappings": self.total_mappings,
            "high_confidence_mappings": self.high_confidence_mappings,
            "domain_similarity": self.domain_similarity,
            "metadata": self.metadata,
        }


@dataclass
class TransferConfig:
    """Configuration for transfer service."""
    similarity_threshold: float = 0.6
    discount_factor: float = 0.8  # Discount for domain differences
    min_confidence: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_historical_data: bool = True
    max_mappings_per_pair: int = 100
    enable_hierarchical_detection: bool = True
    enable_alias_detection: bool = True
    cache_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "discount_factor": self.discount_factor,
            "min_confidence": self.min_confidence,
            "embedding_model": self.embedding_model,
            "use_historical_data": self.use_historical_data,
            "max_mappings_per_pair": self.max_mappings_per_pair,
            "enable_hierarchical_detection": self.enable_hierarchical_detection,
            "enable_alias_detection": self.enable_alias_detection,
            "cache_embeddings": self.cache_embeddings,
        }


@dataclass
class TransferAnalysisResult:
    """Result of transfer analysis."""
    transfers: List[DocumentTransfer]
    total_documents: int
    total_mappings: int
    average_transfer_score: float
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transfers": [t.to_dict() for t in self.transfers],
            "total_documents": self.total_documents,
            "total_mappings": self.total_mappings,
            "average_transfer_score": self.average_transfer_score,
            "computation_time_ms": self.computation_time_ms,
            "metadata": self.metadata,
        }
