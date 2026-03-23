from typing import Optional

from ..builder import BaseDocumentEngine
from ..repository import BaseDocumentRepository
from ..parser import ParserFactory
from ..models import Document, DocumentTree


class DocumentManager:
    """Manages document operations"""

    def __init__(self, 
                 repository: BaseDocumentRepository,
                 document_engine: BaseDocumentEngine
                 ):
        self.parser_factory = ParserFactory()
        self.repository = repository
        self.engine = document_engine
        self.documents = {}

    def load_document(self, file_path: str) -> str:
        """Load document from file"""
        parser = self.parser_factory.create_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for {file_path}")

        document = parser.parse(file_path)

        # Store in memory and repository
        self.documents[document.id] = document
        self.repository.save_document(document, document.id)

        return document.id

    def load_parsed_document(self, json_path: str) -> str:
        """Load pre-parsed document from JSON"""
        parser = self.parser_factory.create_pdf_parser()
        document = parser.from_json(json_path)

        if not document:
            raise ValueError("Failed to load document from JSON")

        if document.id in self.documents:
            raise ValueError(f"Document with ID {document.id} already loaded")

        self.documents[document.id] = document
        self.repository.save_document(document, document.id)

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

    def build_document_tree(self, doc_id: str, target_query: Optional[str] = None) -> DocumentTree:
        """Build document tree"""
        document = self.get_document(doc_id)
        if not document:
            raise ValueError(f"Document {doc_id} not found")

        tree = self.engine.ingest_and_map(document, target_query)

        # Save tree
        self.repository.save_tree(tree, doc_id)

        return tree

    def get_document_tree(self, doc_id: str) -> Optional[DocumentTree]:
        """Get document tree by ID"""
        return self.repository.get_tree(doc_id)

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

    def load_and_build(self, file_path: str, target_query: Optional[str] = None) -> DocumentTree:
        """Load document and build tree in one step"""
        doc_id = self.load_document(file_path)
        return self.build_document_tree(doc_id, target_query)
