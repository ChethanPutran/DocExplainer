from pathlib import Path
from typing import Optional

from doc_explainer.core.document.builder.document_tree_builder import DocumentTreeBuilder

from .pipelines.document import doc_ingestion_pipeline

from .builder import BaseDocumentEngine
from ...store.document.repository import BaseDocumentRepository
from .parser import ParserFactory
from .models import Document, DocumentTree



class DocumentManager:
    """Manages document operations"""

    def __init__(self, 
                 repository: BaseDocumentRepository,
                 document_engine: BaseDocumentEngine,
                 document_tree_builder: DocumentTreeBuilder
                 ):
        self.parser_factory = ParserFactory()
        self.repository = repository
        self.engine = document_engine
        self.document_tree_builder = document_tree_builder
        self.documents = {}

    def load_document(self, file_path: str) -> str:
        """Load document from file"""
        parser = self.parser_factory.create_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for {file_path}")

        metadata = parser.parse_metadata(file_path)
        document = Document(
            path=Path(file_path),
            title=metadata.title or metadata.filename,
            metadata={
                **metadata.metadata,
                "document_id": metadata.document_id,
                "author": metadata.author,
                "page_count": metadata.page_count,
                "file_size": metadata.file_size,
            },
        )
        doc_id = document.id

        if not self.repository.save_document_model(document, doc_id):
            raise ValueError(f"Failed to save parsed document {file_path}")

        self.documents[doc_id] = document

        return doc_id


    def load_parsed_document(self, json_path: str) -> str:
        """Load pre-parsed document from JSON"""
        parser = self.parser_factory.create_pdf_parser()
        document = parser.from_json(json_path)

        if not document:
            raise ValueError("Failed to load document from JSON")

        if document.id in self.documents:
            raise ValueError(f"Document with ID {document.id} already loaded")

        self.documents[document.id] = document
        self.repository.save_document(document.path, document.id)

        return document.id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        # Check memory first
        if doc_id in self.documents:
            return self.documents[doc_id]

        # Check repository
        document = self.repository.get_document(doc_id)
        if document:
            self.documents[doc_id] = document

        return document

    def build_document_tree(self, doc_id: str) -> DocumentTree:
        """Build document tree"""
        document = self.get_document(doc_id)
        if not document:
            raise ValueError(f"Document {doc_id} not found")

        tree = self.document_tree_builder.build(document.id)

        # Save tree
        self.repository.save_document_tree(tree, doc_id)

        return tree

    def get_document_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Get document tree by ID"""
        return self.repository.get_document_tree(doc_id)

    def save_parsed_document(self, doc_id: str, output_path: str):
        """Save document to JSON"""
        document = self.get_document(doc_id)
        if not document:
            raise ValueError(f"Document {doc_id} not found")

        parser = self.parser_factory.create_pdf_parser()
        parser.to_json(document, output_path)

    def list_documents(self) -> list:
        """List all document IDs"""
        return self.repository.list_documents()

    def has_document(self, doc_id: str) -> bool:
        """Check if document exists"""
        return doc_id in self.documents or self.repository.get_document(doc_id) is not None


    def process_document(self, document_id: str, target_query: str = None, force_reprocess: bool = False) -> str:
        """Orchestrate full processing via myflow."""
        document = self.get_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        id = self.engine.ingest(document.path)  # Ingest document into the engine
        
        # Store in memory and repository
        self.documents[document.id] = document

        # # Run pipeline
        # run_id = doc_ingestion_pipeline.run(
        #     file_path=document.path,
        #     target_query=target_query,
        #     force_reprocess=force_reprocess
        # )

        # Mark in registry
        # We need the doc_id; we can get it from the pipeline result if we wait.
        # For async, we can store later.
        # We'll retrieve the result after run.
        # Since myflow run returns only run_id, we need to fetch result separately.
        # We'll implement a `get_result` method in Pipeline.
        # For now, we'll assume we can get the doc_id from the registry after processing.
        # We'll just return run_id.
        return id