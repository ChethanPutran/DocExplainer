from abc import ABC, abstractmethod
from typing import Optional
from ..models.structure import Document
from ..models.tree import DocumentTree


class BaseDocumentEngine(ABC):
    """Base interface for document engine"""

    @abstractmethod
    def ingest_and_map(self, document: Document, target_query: Optional[str] = None) -> DocumentTree:
        """
        Full processing pipeline:
        1. Full indexing
        2. Target discovery
        3. Tree building
        4. Tree indexing
        """
        pass

    @abstractmethod
    def query(self, user_query: str, level: str = "paragraph", k: int = 3) -> list:
        """Search within hierarchical summaries"""
        pass

    @abstractmethod
    def get_document_tree(self) -> Optional[DocumentTree]:
        """Get the built document tree"""
        pass


class DocumentBuilder(ABC):
    langchain_embeddings = False  # Flag to indicate if LangChain embeddings are used
    """Base interface for document builders"""
    @abstractmethod
    def build_tree(self, document: Document, target_section: Optional[str] = None) -> DocumentTree:
        """Build document tree with summaries"""
        pass

    @abstractmethod
    def create_full_vector_db(self, document: Document, collection_name: str = "full_doc",
                              persist_directory: Optional[str] = None)-> object:
        """Create full-document vector database"""
        pass

    @abstractmethod
    def create_tree_aware_db(self, tree: DocumentTree, collection_name: str = "hierarchical_db",
                             persist_directory: Optional[str] = None) -> object:
        """Create vector database from tree chunks"""
        pass

    @abstractmethod
    def visualize_tree(self, node, indent: str = "", is_last: bool = True):
        """Simple text visualization of the tree structure"""
        pass
