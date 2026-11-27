"""The composition root: wires the core packages together. Nothing else may."""
from .ingestion import EmptyDocumentError, IngestReport, IngestionPipeline
from .system import System, SystemConfig, build_system

__all__ = ["System", "SystemConfig", "build_system", "IngestionPipeline", "IngestReport", "EmptyDocumentError"]
