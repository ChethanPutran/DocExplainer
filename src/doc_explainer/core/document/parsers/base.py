"""The parser contract and the registry that picks a parser by file extension.

A parser's only job is to turn a file into an ordered stream of ``Block``s (see
``core.common.structure``). It knows nothing about sections or hierarchy: that is the
``StructureBuilder``'s job, done once for every format. Adding a format = one parser class and one
``register`` call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Protocol, Tuple, runtime_checkable

from ...common.structure import ParsedDocument


class UnsupportedFormat(ValueError):
    """No parser is registered for this file's extension."""


class ParseError(ValueError):
    """The file could not be read as the format its extension claims."""


@runtime_checkable
class Parser(Protocol):
    extensions: Tuple[str, ...]

    def parse(self, path: Path) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self) -> None:
        self._by_ext: Dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        for ext in parser.extensions:
            self._by_ext[ext.lower()] = parser

    def for_path(self, path) -> Parser:
        ext = Path(path).suffix.lower()
        try:
            return self._by_ext[ext]
        except KeyError:
            raise UnsupportedFormat(
                f"No parser for '{ext or Path(path).name}'. Supported: {', '.join(self.supported_extensions())}")

    def supported_extensions(self) -> List[str]:
        return sorted(self._by_ext)

    def parse(self, path) -> ParsedDocument:
        path = Path(path)
        parser = self.for_path(path)
        try:
            return parser.parse(path)
        except (UnsupportedFormat, ParseError):
            raise
        except Exception as e:  # a corrupt or mislabelled file must be a clean, reportable error
            raise ParseError(f"Could not read {path.name} as {path.suffix.lower()}: {e}") from e
