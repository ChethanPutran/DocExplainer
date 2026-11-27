"""IngestionPipeline: file -> structure -> index -> knowledge, as one plain sequence with progress and cancellation.

    parse (any format)      -> unified DocumentStructure
    index                   -> source text, structure graph (no text), embeddings in the vector index
    knowledge (per section) -> the temporal knowledge graph

A document that cannot be read at all is an error. A failure of the knowledge stage alone is not: the
document is already searchable, so it is kept and the problem reported, and the build can be resumed later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..core.common.structure import DocumentStructure, NodeKind
from ..core.document import ParserRegistry, StructureBuilder, default_registry
from ..core.indexing import Indexer
from ..core.knowledge import KnowledgeBuildError, TemporalKnowledgeBuilder

logger = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


class EmptyDocumentError(ValueError):
    """The file parsed, but contains no readable text (for example a scanned PDF with no text layer)."""


@dataclass
class IngestReport:
    doc_id: str
    title: str
    format: str
    pages: int
    sections: int
    chunks: int
    concepts: int = 0
    relations: int = 0
    knowledge: str = "none"          # ready | partial | failed | none (not requested)
    knowledge_error: str = ""


class IngestionPipeline:
    def __init__(self, indexer: Indexer, knowledge: Optional[TemporalKnowledgeBuilder] = None,
                 registry: Optional[ParserRegistry] = None, builder: Optional[StructureBuilder] = None):
        self.indexer, self.knowledge = indexer, knowledge
        self.registry, self.builder = registry or default_registry(), builder or StructureBuilder()

    def supported_extensions(self):
        return self.registry.supported_extensions()

    def parse(self, path, doc_id: str) -> DocumentStructure:
        structure = self.builder.build(self.registry.parse(Path(path)), doc_id)
        if not any(n.text.strip() for n in structure.nodes if n.kind is NodeKind.PARAGRAPH):
            raise EmptyDocumentError("No readable text was found in this file. If it is a scanned document, it needs OCR first.")
        return structure

    def run(self, path, doc_id: str, build_knowledge: bool = True, progress: Optional[Progress] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> IngestReport:
        say = progress or (lambda stage, fraction: None)
        say("reading the file", 0.0)
        structure = self.parse(path, doc_id)
        say("reading the file", 0.05)
        report_index = self.indexer.index(structure, progress=lambda stage, f: say(stage, 0.05 + 0.30 * f))
        report = IngestReport(doc_id=doc_id, title=structure.title, format=structure.source_format, pages=structure.page_count,
                              sections=report_index.sections, chunks=report_index.chunks)
        if build_knowledge and self.knowledge is not None:
            try:
                kr = self.knowledge.build(structure, progress=lambda stage, f: say(stage, 0.35 + 0.65 * f), cancelled=cancelled)
                report.concepts, report.relations = kr.concepts, kr.relations
                report.knowledge = "partial" if kr.steps_failed else "ready"
                report.knowledge_error = kr.errors[-1] if kr.errors else ""
            except KnowledgeBuildError as e:
                logger.warning("Knowledge build failed for %s: %s", doc_id, e)
                report.knowledge, report.knowledge_error = "failed", str(e)
        say("done", 1.0)
        return report

    def remove(self, doc_id: str) -> None:
        self.indexer.remove(doc_id)
        if self.knowledge is not None:
            self.knowledge.store.reset(doc_id)
