"""Backward-compatible exports for graph-related knowledge-modelling components.

This module used to contain all graph logic in a single file.
It now re-exports modular implementations from `src.core.knowlege_modelling.graph`.
"""

from src.core.knowlege_modelling.graph import (
    ConceptBuilder,
    DocumentChain,
    GraphStateManager,
    GraphUpdater,
    build_document_hierarchy,
)

__all__ = [
    "ConceptBuilder",
    "DocumentChain",
    "GraphStateManager",
    "GraphUpdater",
    "build_document_hierarchy",
]
