"""Multi-document knowledge transfer service."""

import time
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from src.core.knowledge.models.transfer_models import (
        ConceptMapping,
        DocumentTransfer,
        TransferConfig,
        TransferAnalysisResult,
        ConceptAlignmentType,
    )
except ImportError:
    # Fallback for relative imports
    from .models.transfer_models import (
        ConceptMapping,
        DocumentTransfer,
        TransferConfig,
        TransferAnalysisResult,
        ConceptAlignmentType,
    )

logger = logging.getLogger(__name__)


class ConceptSimilarityMatrix:
    """Manages semantic similarity calculations between concepts."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize similarity matrix calculator.
        
        Args:
            model_name: HuggingFace model for embeddings
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required")
        
        self.model = SentenceTransformer(model_name)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get embedding for text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector
        """
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]

        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if use_cache:
            self.embedding_cache[text] = embedding
        
        return embedding

    def compute_similarity(
        self, text1: str, text2: str, use_cache: bool = True
    ) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            use_cache: Whether to use cached results
            
        Returns:
            Similarity score (0-1)
        """
        cache_key = (text1, text2)
        if use_cache and cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        emb1 = self.get_embedding(text1, use_cache)
        emb2 = self.get_embedding(text2, use_cache)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity = float(np.clip(similarity, 0, 1))

        if use_cache:
            self.similarity_cache[cache_key] = similarity

        return similarity

    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.similarity_cache.clear()


class ManualMappingStore:
    """Stores and manages manual concept mappings."""

    def __init__(self):
        """Initialize manual mapping store."""
        self.mappings: Dict[Tuple[str, str, str, str], Dict] = {}

    def add_mapping(
        self,
        source_concept: str,
        target_concept: str,
        source_doc: str,
        target_doc: str,
        alignment_type: ConceptAlignmentType = ConceptAlignmentType.EQUIVALENT,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a manual mapping.
        
        Args:
            source_concept: Source concept
            target_concept: Target concept
            source_doc: Source document
            target_doc: Target document
            alignment_type: Type of alignment
            metadata: Additional metadata
        """
        key = (source_doc, target_doc, source_concept, target_concept)
        self.mappings[key] = {
            "alignment_type": alignment_type,
            "metadata": metadata or {},
        }

    def get_mapping(
        self,
        source_concept: str,
        target_concept: str,
        source_doc: str,
        target_doc: str,
    ) -> Optional[Dict]:
        """Get a manual mapping."""
        key = (source_doc, target_doc, source_concept, target_concept)
        return self.mappings.get(key)

    def get_mappings_for_doc_pair(
        self, source_doc: str, target_doc: str
    ) -> List[Dict]:
        """Get all mappings for a document pair."""
        return [
            v
            for (s_doc, t_doc, _, _), v in self.mappings.items()
            if s_doc == source_doc and t_doc == target_doc
        ]


class ConceptAlignmentDetector:
    """Detects aligned concepts across documents."""

    def __init__(self, similarity_matrix: ConceptSimilarityMatrix):
        """Initialize detector.
        
        Args:
            similarity_matrix: Similarity calculator
        """
        self.similarity_matrix = similarity_matrix

    def detect_identical_concepts(
        self, concepts1: List[str], concepts2: List[str]
    ) -> List[Tuple[str, str, float]]:
        """Detect identical or exact-match concepts.
        
        Args:
            concepts1: First set of concepts
            concepts2: Second set of concepts
            
        Returns:
            List of (concept1, concept2, score) tuples
        """
        identical = []
        
        concepts1_lower = {c.lower(): c for c in concepts1}
        concepts2_lower = {c.lower(): c for c in concepts2}
        
        for c1_lower, c1_orig in concepts1_lower.items():
            if c1_lower in concepts2_lower:
                identical.append((c1_orig, concepts2_lower[c1_lower], 1.0))
        
        return identical

    def detect_alias_concepts(
        self,
        concepts1: List[str],
        concepts2: List[str],
        aliases_map: Optional[Dict[str, List[str]]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Detect concepts that are aliases of each other.
        
        Args:
            concepts1: First set of concepts
            concepts2: Second set of concepts
            aliases_map: Mapping of concept to aliases
            
        Returns:
            List of (concept1, concept2, score) tuples
        """
        if not aliases_map:
            return []

        aliases = []
        concepts2_set = set(c.lower() for c in concepts2)

        for c1 in concepts1:
            if c1 not in aliases_map:
                continue

            for alias in aliases_map[c1]:
                if alias.lower() in concepts2_set:
                    aliases.append((c1, alias, 0.95))

        return aliases

    def detect_hierarchical_relationships(
        self,
        concepts1: List[str],
        concepts2: List[str],
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[str, str, float, str]]:
        """Detect hierarchical (parent-child) relationships.
        
        Args:
            concepts1: First set of concepts
            concepts2: Second set of concepts
            similarity_threshold: Minimum similarity
            
        Returns:
            List of (concept1, concept2, score, relation_type) tuples
        """
        relationships = []

        for c1 in concepts1:
            for c2 in concepts2:
                sim = self.similarity_matrix.compute_similarity(c1, c2)

                if sim < similarity_threshold:
                    continue

                # Heuristic: shorter concept often is parent/more general
                relation = "parent" if len(c1) < len(c2) else "child"
                relationships.append((c1, c2, sim, relation))

        return relationships

    def detect_similar_concepts(
        self,
        concepts1: List[str],
        concepts2: List[str],
        similarity_threshold: float = 0.6,
    ) -> List[Tuple[str, str, float]]:
        """Detect semantically similar concepts.
        
        Args:
            concepts1: First set of concepts
            concepts2: Second set of concepts
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (concept1, concept2, similarity_score) tuples
        """
        similar = []
        
        for c1 in concepts1:
            for c2 in concepts2:
                sim = self.similarity_matrix.compute_similarity(c1, c2)
                
                if sim >= similarity_threshold:
                    similar.append((c1, c2, sim))
        
        return similar


class TransferEffectivenessEstimator:
    """Estimates effectiveness of knowledge transfer."""

    def __init__(self, config: TransferConfig):
        """Initialize estimator.
        
        Args:
            config: Transfer configuration
        """
        self.config = config
        self.historical_data: Dict[str, float] = {}

    def estimate_transfer_effectiveness(
        self,
        source_concept: str,
        target_concept: str,
        source_doc: str,
        target_doc: str,
        similarity_score: float,
        domain_similarity: float = 0.5,
    ) -> float:
        """Estimate transfer effectiveness (0-1).
        
        Args:
            source_concept: Source concept
            target_concept: Target concept
            source_doc: Source document
            target_doc: Target document
            similarity_score: Semantic similarity (0-1)
            domain_similarity: Domain similarity discount (0-1)
            
        Returns:
            Transfer effectiveness score (0-1)
        """
        # Base effectiveness on similarity
        effectiveness = similarity_score

        # Apply domain similarity discount
        effectiveness *= domain_similarity * self.config.discount_factor

        # Apply historical success multiplier if available
        if self.config.use_historical_data:
            history_key = f"{source_concept}->{target_concept}"
            if history_key in self.historical_data:
                avg_historical = self.historical_data[history_key]
                # Weight: 70% current, 30% historical
                effectiveness = 0.7 * effectiveness + 0.3 * avg_historical

        return float(np.clip(effectiveness, 0, 1))

    def estimate_domain_similarity(
        self, doc1_concepts: List[str], doc2_concepts: List[str]
    ) -> float:
        """Estimate similarity between document domains.
        
        Args:
            doc1_concepts: Concepts from document 1
            doc2_concepts: Concepts from document 2
            
        Returns:
            Domain similarity score (0-1)
        """
        # Jaccard similarity of concept sets
        set1 = set(c.lower() for c in doc1_concepts)
        set2 = set(c.lower() for c in doc2_concepts)

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def record_transfer_success(self, transfer_key: str, success_rate: float) -> None:
        """Record historical transfer success.
        
        Args:
            transfer_key: Key identifying the transfer
            success_rate: Success rate (0-1)
        """
        if transfer_key in self.historical_data:
            # Update with moving average
            self.historical_data[transfer_key] = (
                0.7 * self.historical_data[transfer_key] + 0.3 * success_rate
            )
        else:
            self.historical_data[transfer_key] = success_rate


class TransferService:
    """Manages multi-document knowledge transfer."""

    def __init__(self, config: Optional[TransferConfig] = None):
        """Initialize transfer service.
        
        Args:
            config: Transfer configuration
        """
        self.config = config or TransferConfig()
        self.similarity_matrix = ConceptSimilarityMatrix(self.config.embedding_model)
        self.alignment_detector = ConceptAlignmentDetector(self.similarity_matrix)
        self.effectiveness_estimator = TransferEffectivenessEstimator(self.config)
        self.manual_mappings = ManualMappingStore()
        self.db_manager = None

    def set_db_manager(self, db_manager):
        """Set database manager for persistence.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    def find_concept_mappings(
        self,
        source_doc: str,
        target_doc: str,
        source_concepts: List[str],
        target_concepts: List[str],
    ) -> List[ConceptMapping]:
        """Find mappings between concepts in two documents.
        
        Args:
            source_doc: Source document ID
            target_doc: Target document ID
            source_concepts: Concepts from source document
            target_concepts: Concepts from target document
            
        Returns:
            List of concept mappings
        """
        mappings = []
        matched_targets = set()

        # 1. Detect identical concepts
        identical = self.alignment_detector.detect_identical_concepts(
            source_concepts, target_concepts
        )
        for src, tgt, sim in identical:
            mapping = ConceptMapping(
                source_concept=src,
                target_concept=tgt,
                source_doc=source_doc,
                target_doc=target_doc,
                similarity_score=sim,
                transfer_score=0.95,
                alignment_type=ConceptAlignmentType.IDENTICAL,
                confidence=0.95,
            )
            mappings.append(mapping)
            matched_targets.add(tgt)

        # 2. Detect aliases (if configured)
        if self.config.enable_alias_detection:
            aliases = self.alignment_detector.detect_alias_concepts(
                source_concepts, target_concepts
            )
            for src, tgt, sim in aliases:
                if tgt not in matched_targets:
                    mapping = ConceptMapping(
                        source_concept=src,
                        target_concept=tgt,
                        source_doc=source_doc,
                        target_doc=target_doc,
                        similarity_score=sim,
                        transfer_score=0.9,
                        alignment_type=ConceptAlignmentType.ALIAS,
                        confidence=0.9,
                    )
                    mappings.append(mapping)
                    matched_targets.add(tgt)

        # 3. Detect hierarchical relationships (if configured)
        if self.config.enable_hierarchical_detection:
            hierarchical = self.alignment_detector.detect_hierarchical_relationships(
                source_concepts,
                target_concepts,
                self.config.similarity_threshold,
            )
            for src, tgt, sim, relation in hierarchical:
                if tgt not in matched_targets:
                    mapping = ConceptMapping(
                        source_concept=src,
                        target_concept=tgt,
                        source_doc=source_doc,
                        target_doc=target_doc,
                        similarity_score=sim,
                        transfer_score=sim * 0.8,
                        alignment_type=ConceptAlignmentType.HIERARCHICAL,
                        confidence=sim,
                        metadata={"relation": relation},
                    )
                    mappings.append(mapping)
                    matched_targets.add(tgt)

        # 4. Detect similar concepts
        similar = self.alignment_detector.detect_similar_concepts(
            source_concepts,
            target_concepts,
            self.config.similarity_threshold,
        )
        for src, tgt, sim in similar:
            if (
                tgt not in matched_targets
                and sim >= self.config.similarity_threshold
            ):
                mapping = ConceptMapping(
                    source_concept=src,
                    target_concept=tgt,
                    source_doc=source_doc,
                    target_doc=target_doc,
                    similarity_score=sim,
                    transfer_score=sim,
                    alignment_type=ConceptAlignmentType.SIMILAR,
                    confidence=sim,
                )
                mappings.append(mapping)
                matched_targets.add(tgt)

        # Limit results
        mappings = mappings[: self.config.max_mappings_per_pair]

        return mappings

    def analyze_document_pair_transfer(
        self,
        source_doc: str,
        target_doc: str,
        source_concepts: List[str],
        target_concepts: List[str],
    ) -> DocumentTransfer:
        """Analyze transfer between two documents.
        
        Args:
            source_doc: Source document ID
            target_doc: Target document ID
            source_concepts: Concepts from source
            target_concepts: Concepts from target
            
        Returns:
            Document transfer analysis
        """
        # Find mappings
        mappings = self.find_concept_mappings(
            source_doc, target_doc, source_concepts, target_concepts
        )

        # Estimate domain similarity
        domain_sim = self.effectiveness_estimator.estimate_domain_similarity(
            source_concepts, target_concepts
        )

        # Estimate transfer effectiveness for each mapping
        for mapping in mappings:
            effectiveness = (
                self.effectiveness_estimator.estimate_transfer_effectiveness(
                    mapping.source_concept,
                    mapping.target_concept,
                    source_doc,
                    target_doc,
                    mapping.similarity_score,
                    domain_sim,
                )
            )
            mapping.transfer_score = effectiveness

        # Calculate overall score
        if mappings:
            overall_score = np.mean([m.transfer_score for m in mappings])
        else:
            overall_score = 0.0

        transfer = DocumentTransfer(
            source_doc=source_doc,
            target_doc=target_doc,
            concept_mappings=mappings,
            overall_score=float(overall_score),
            domain_similarity=domain_sim,
        )

        return transfer

    def analyze_multi_document_transfer(
        self, documents: Dict[str, List[str]]
    ) -> TransferAnalysisResult:
        """Analyze knowledge transfer across multiple documents.
        
        Args:
            documents: Dict mapping document ID to concept list
            
        Returns:
            Transfer analysis result
        """
        start_time = time.time()
        transfers = []
        all_mappings = []

        doc_ids = list(documents.keys())

        # Analyze all document pairs
        for i, source_doc in enumerate(doc_ids):
            for target_doc in doc_ids[i + 1 :]:
                transfer = self.analyze_document_pair_transfer(
                    source_doc,
                    target_doc,
                    documents[source_doc],
                    documents[target_doc],
                )
                transfers.append(transfer)
                all_mappings.extend(transfer.concept_mappings)

        # Calculate statistics
        total_mappings = len(all_mappings)
        avg_score = (
            np.mean([m.transfer_score for m in all_mappings])
            if all_mappings
            else 0.0
        )

        computation_time_ms = (time.time() - start_time) * 1000

        result = TransferAnalysisResult(
            transfers=transfers,
            total_documents=len(documents),
            total_mappings=total_mappings,
            average_transfer_score=float(avg_score),
            computation_time_ms=computation_time_ms,
        )

        return result

    def save_transfer_mappings(self, transfer: DocumentTransfer) -> None:
        """Save transfer mappings to database.
        
        Args:
            transfer: Document transfer to save
        """
        if not self.db_manager:
            logger.warning("No database manager configured")
            return

        for mapping in transfer.concept_mappings:
            try:
                self.db_manager.execute(
                    """
                    INSERT OR REPLACE INTO concept_transfer
                    (source_doc, target_doc, source_concept, target_concept, transfer_score)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        mapping.source_doc,
                        mapping.target_doc,
                        mapping.source_concept,
                        mapping.target_concept,
                        mapping.transfer_score,
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to save mapping: {e}")

    def add_manual_mapping(
        self,
        source_concept: str,
        target_concept: str,
        source_doc: str,
        target_doc: str,
        alignment_type: ConceptAlignmentType = ConceptAlignmentType.EQUIVALENT,
    ) -> None:
        """Add a manual concept mapping.
        
        Args:
            source_concept: Source concept
            target_concept: Target concept
            source_doc: Source document
            target_doc: Target document
            alignment_type: Type of alignment
        """
        self.manual_mappings.add_mapping(
            source_concept,
            target_concept,
            source_doc,
            target_doc,
            alignment_type,
        )
