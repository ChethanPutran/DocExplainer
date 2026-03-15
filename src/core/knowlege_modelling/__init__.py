"""Backward-compatible exports for knowledge-modelling components. --- IGNORE ---
This module serves as a central export point for all knowledge-modelling related components, including graph structures, user knowledge tracing, and concept extraction. --- IGNORE ---
It re-exports modular implementations from various submodules within `src.core.knowlege_modelling`. --- IGNORE ---
"""

from core.knowlege_modelling.graph.base import (
    Concept,
    ConceptGraph,
    ConceptNode,
    ConceptNodeRelationship,
    ConceptRelationship,
    GraphDelta,
)
from src.core.knowlege_modelling.graph import (
    ConceptBuilder,
    DocumentChain,
    GraphStateManager,
    GraphUpdater,
    build_document_hierarchy,
)
from core.knowlege_modelling.user.knowledge_tracing import BayesianKnowledgeTracer
from core.knowlege_modelling.extraction.extraction import RelationshipExtractor,ConceptExtractor
from core.knowlege_modelling.user.base import KnowledgeState, UserKnowledgeState
from core.knowlege_modelling.user.user_manager import UserManager

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
