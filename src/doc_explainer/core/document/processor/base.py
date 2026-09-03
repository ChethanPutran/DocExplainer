from abc import ABC, abstractmethod
from pathlib import Path
from ..models.structure import Section
from ..models.base import ProcessedSection, ProcessingContext

class DocumentProcessor(ABC):
    """Base interface for document builders"""

    @abstractmethod
    def visualize_tree(self, node, indent: str = "", is_last: bool = True):
        """Simple text visualization of the tree structure"""
        pass

    @abstractmethod
    def process(self, section: Section, context: ProcessingContext) -> ProcessedSection:
        """Process the document and return a DocumentTree"""
        pass
