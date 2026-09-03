from pathlib import Path
from typing import Optional

from ...store.checkpoint.base import CheckpointStore
from ...store.graph.base import GraphStore
from ...store.vector.base import VectorStore
from .models.base import ProcessingContext
from .parser.base import DocumentParser
from .processor.base import DocumentProcessor
from .builder.base import BaseDocumentEngine


class DocumentEngine(BaseDocumentEngine):
    """Orchestrates document processing pipeline"""
    
    def __init__(self, 
                 parser:DocumentParser,
                 processor: DocumentProcessor,
                 vector_store: VectorStore,
                 graph_store: GraphStore,
                 checkpoint_store: Optional[CheckpointStore] = None):
        self.parser = parser
        self.checkpoint_store = checkpoint_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.processor = processor

    def ingest(self, file_path: Path, target_query: Optional[str] = None) -> str:
        """Ingest document from file path and process"""
        # ------------------------------------
        # 1. Parse metadata
        # ------------------------------------
        assert self.parser is not None, "Parser must be set before ingestion"

        document = self.parser.parse_metadata(
            file_path
        )
        if(not document):
            raise ValueError(f"Failed to parse document metadata from {file_path}")
        
        document_id = document.document_id

        if self.checkpoint_store:
            if self.checkpoint_store.is_run_complete(document_id):
                return document_id

            self.checkpoint_store.start(
                namespace=document_id,
                file_path=str(file_path),
            )

        context = ProcessingContext(
            namespace=document.document_id
        )

        if self.graph_store:
            self.graph_store.add_document(
                document_id=document.document_id,
                title=document.title or document.filename,
                namespace=document.document_id,
                metadata={
                    "title": document.title,
                    "filename": document.filename,
                    "page_count": document.page_count,
                },
            )

        # --------------------------------------------------------------
        # 2. Stream sections
        # --------------------------------------------------------------

        for index, section in enumerate(self.parser.iter_sections(file_path)):
            try:
                context.section_index = index

                # --------------------------------
                # Resume support
                # --------------------------------
                if (
                    self.checkpoint_store
                    and self.checkpoint_store.is_completed(
                        document.document_id,
                        section.id
                    )
                ):
                    continue

                if self.checkpoint_store:
                    self.checkpoint_store.mark_started(
                        document.document_id,
                        section.id
                    )

                # ------------------------------------------------------
                # Process only current section
                # ------------------------------------------------------

                processed_section =  self.processor.process(
                        section,
                        context
                    )

                vector_documents = list(
                    processed_section.vector_documents()
                )

                # --------------------------------------------------
                # Stream vectors into vector DB
                # --------------------------------------------------

                if self.vector_store:
                    self.vector_store.add(
                        namespace=document.document_id,
                        documents=iter(vector_documents)
                    )

                
                # --------------------------------------------------
                # Stream relationships into graph
                # --------------------------------------------------

                if self.graph_store:
                    self.graph_store.add_section(
                        namespace=document.document_id,
                        section=processed_section
                    )
                    for vector_document in vector_documents:
                        if vector_document.metadata.get("level") != "section":
                            self.graph_store.add_chunk(
                                namespace=document.document_id,
                                chunk_id=vector_document.id,
                                text=vector_document.text,
                                metadata=vector_document.metadata,
                            )
                    self.graph_store.add_relationships(
                        namespace=document.document_id,
                        relationships=processed_section.relationships(),
                    )

                # --------------------------------
                # Checkpoint
                # --------------------------------

                if self.checkpoint_store:
                    self.checkpoint_store.mark_completed(
                        namespace=document.document_id,
                        section_id=section.id
                    )

                # --------------------------------
                # Update context
                # --------------------------------
                context.previous_section_summary = (
                    processed_section.summary
                )

                # Release section
                del processed_section
                del section
            except Exception as exc:
                if self.checkpoint_store:
                    self.checkpoint_store.mark_section_failed(
                        namespace=document_id,
                        section_id=section.id,
                        error=str(exc),
                    )
                raise exc

        if self.checkpoint_store:
            self.checkpoint_store.complete(
                namespace=document_id,
            )
        return document.document_id

