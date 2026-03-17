from typing import Optional
from .base import DocumentParser
from .pdf_parser import PDFParser


class ParserFactory:
    """Factory for creating document parsers"""
    
    _parsers = {
        '.pdf': PDFParser,
    }
    
    @classmethod
    def register_parser(cls, extension: str, parser_class):
        """Register a new parser for an extension"""
        cls._parsers[extension.lower()] = parser_class
    
    @classmethod
    def create_parser(cls, file_path: str, **kwargs) -> Optional[DocumentParser]:
        """Create appropriate parser for file"""
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        parser_class = cls._parsers.get(ext)
        if parser_class:
            return parser_class(**kwargs)
        
        return None
    
    @classmethod
    def create_pdf_parser(cls, **kwargs) -> PDFParser:
        """Create PDF parser"""
        return PDFParser(**kwargs)