"""Parse any supported document into the unified structure."""
from pathlib import Path
from typing import Optional, Union

from ..common.structure import DocumentStructure
from .builder import StructureBuilder
from .parsers import ParseError, ParserRegistry, UnsupportedFormat, default_registry


def parse_document(path: Union[str, Path], doc_id: str, registry: Optional[ParserRegistry] = None,
                   builder: Optional[StructureBuilder] = None) -> DocumentStructure:
    """File -> parser chosen by extension -> blocks -> unified ``DocumentStructure``."""
    registry = registry or default_registry()
    builder = builder or StructureBuilder()
    return builder.build(registry.parse(path), doc_id)


__all__ = ["parse_document", "StructureBuilder", "default_registry", "ParserRegistry", "UnsupportedFormat", "ParseError"]
