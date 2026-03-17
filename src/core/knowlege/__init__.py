"""Backward-compatible exports for knowledge-modelling components. --- IGNORE ---
This module serves as a central export point for all knowledge-modelling related components, including graph structures, user knowledge tracing, and concept extraction. --- IGNORE ---
It re-exports modular implementations from various submodules within `src.core.knowlege_modelling`. --- IGNORE ---
"""

from src.core.knowlege.base import (
    Concept,
    ConceptGraph,
    ConceptNode,
    ConceptNodeRelationship,
    ConceptRelationship,
    GraphDelta,
)
from src.core.knowlege.graph import (
    ConceptBuilder,
    DocumentChain,
    GraphStateManager,
    GraphUpdater,
    build_document_hierarchy,
)
from src.core.knowlege.user.knowledge_tracing import BayesianKnowledgeTracer
from core.knowlege.extraction.extractor import RelationshipExtractor,ConceptExtractor
from src.core.knowlege.user.base import KnowledgeState, UserKnowledgeState
from src.core.knowlege.user.user_manager import UserManager

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
