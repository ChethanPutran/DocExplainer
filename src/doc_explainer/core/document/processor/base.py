from abc import ABC, abstractmethod
from typing import Optional, Protocol

from ..builder.base import SimilaritySearchDB
from ..models.structure import Section
from doc_explainer.core.document.models.base import ProcessedSection, ProcessingContext
from ..models.structure import Document
from ..models.tree import DocumentTree
from typing import Optional


class SectionProcessor(Protocol):

    def process(
        self,
        section: Section,
        context: "ProcessingContext"
    ) -> ProcessedSection:
        ...

class DocumentProcessor(ABC):
    langchain_embeddings = False  # Flag to indicate if LangChain embeddings are used
    
    """Base interface for document builders"""
    @abstractmethod
    def build_tree(self, document: Document, target_section: Optional[str] = None) -> DocumentTree:
        """Build document tree with summaries"""
        pass

    @abstractmethod
    def create_full_vector_db(self, document: Document, collection_name: str = "full_doc",
                              persist_directory: Optional[str] = None)-> SimilaritySearchDB:
        """Create full-document vector database"""
        pass

    @abstractmethod
    def create_tree_aware_db(self, tree: DocumentTree, collection_name: str = "hierarchical_db",
                             persist_directory: Optional[str] = None) -> SimilaritySearchDB:
        """Create vector database from tree chunks"""
        pass

    @abstractmethod
    def visualize_tree(self, node, indent: str = "", is_last: bool = True):
        """Simple text visualization of the tree structure"""
        pass

    @abstractmethod
    def process(self, section: Section, context: ProcessingContext) -> ProcessedSection:
        """Process the document and return a DocumentTree"""
        pass
