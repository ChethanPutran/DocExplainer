from .base import ParseError, Parser, ParserRegistry, UnsupportedFormat
from .docx import DocxParser
from .html import HtmlParser
from .markdown import MarkdownParser
from .pdf import PdfParser
from .pptx import PptxParser
from .text import TextParser


def default_registry() -> ParserRegistry:
    """Every supported format. To add one: write a parser class and register it here."""
    registry = ParserRegistry()
    for parser in (PdfParser(), HtmlParser(), DocxParser(), MarkdownParser(), TextParser(), PptxParser()):
        registry.register(parser)
    return registry


__all__ = ["default_registry", "ParserRegistry", "Parser", "UnsupportedFormat", "ParseError"]
