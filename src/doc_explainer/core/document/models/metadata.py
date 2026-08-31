from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class Metadata:
    """Metadata for document chunks"""
    length: int = 0
    start: int = 0
    end: int = 0
    raw_text: str = ''
    is_concept: bool = False
    page: int = 0
    font_size: float = 0.0
    font_name: str = ''
    is_bold: bool = False


class MetadataCreator(ABC):
    """Abstract factory for metadata creation"""
    
    @abstractmethod
    def create_metadata(self, **kwargs) -> Metadata:
        """Create metadata object"""
        pass


class SimpleMetadataCreator(MetadataCreator):
    """Simple metadata creator implementation"""
    
    def create_metadata(self, **kwargs) -> Metadata:
        return Metadata(
            length=kwargs.get('length', 0),
            start=kwargs.get('start', 0),
            end=kwargs.get('end', 0),
            raw_text=kwargs.get('raw_text', ''),
            is_concept=kwargs.get('is_concept', False),
            page=kwargs.get('page', 0),
            font_size=kwargs.get('font_size', 0.0),
            font_name=kwargs.get('font_name', ''),
            is_bold=kwargs.get('is_bold', False)
        )