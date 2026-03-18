"""Backward-compatible exports for knowledge-modelling components. --- IGNORE ---
This module serves as a central export point for all knowledge-modelling related components, including graph structures, user knowledge tracing, and concept extraction. --- IGNORE ---
It re-exports modular implementations from various submodules within `src.core.knowlege_modelling`. --- IGNORE ---
"""

from src.core.knowledge.base import (
    Concept,
    ConceptGraph,
    ConceptNode,
    ConceptNodeRelationship,
    ConceptRelationship,
    GraphDelta,
)
from src.core.knowledge.graph import (
    ConceptBuilder,
    DocumentChain,
    GraphStateManager,
    GraphUpdater,
    build_document_hierarchy,
)

from src.core.knowledge.extraction.extractor import RelationshipExtractor,ConceptExtractor
from src.core.knowledge.graph.state_manager import GraphStateManager
from src.core.knowledge.services.prerequisite_analyzer import PrerequisiteAnalyzer
from src.core.knowledge.services.learning_path import LearningPathGenerator
from src.core.knowledge.services.recommendation import RecommendationService
__all__ = [
    "BayesianKnowledgeTracer",
    "Concept",
    "ConceptBuilder",
    "ConceptGraph",
    "ConceptNode",
    "ConceptNodeRelationship",
    "ConceptRelationship",
    "DocumentChain",
    "GraphDelta",
    "GraphStateManager",
    "GraphUpdater",
    "KnowledgeState",
    "RelationshipExtractor",
    "ConceptExtractor",
    "UserKnowledgeState",
    "build_document_hierarchy",
    "UserManager"
]
