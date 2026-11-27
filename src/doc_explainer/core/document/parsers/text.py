"""Plain text: paragraphs are blank-line separated; headings are recognised by common conventions."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ...common.structure import Block, BlockKind, ParsedDocument
from ..sentences import normalize_space

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(\S.{0,120})$")
_CHAPTER = re.compile(r"^(chapter|section|part|appendix)\s+[\dIVXLC]+[.:]?(\s+.{0,100})?$", re.I)
_BULLET = re.compile(r"^\s*([-*•–]|\d+[.)])\s+(\S.*)$")
_UNDERLINE = re.compile(r"^\s*(={3,}|-{3,})\s*$")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _heading_level(line: str, next_line: str) -> int:
    """0 if ``line`` is not a heading, else its level."""
    s = line.strip()
    if not s or len(s) > 120 or s.endswith((".", ",", ";")) or len(s.split()) > 14:
        return 0
    if next_line and _UNDERLINE.match(next_line):
        return 1 if next_line.strip()[0] == "=" else 2
    m = _NUMBERED.match(s)
    if m and m.group(2)[0].isupper() and not re.search(r"[.!?]$", m.group(2)):
        return min(m.group(1).count(".") + 1, 4)
    if _CHAPTER.match(s):
        return 1
    if len(s) >= 4 and s.isupper() and any(c.isalpha() for c in s):
        return 1
    return 0


class TextParser:
    extensions = (".txt", ".text")

    def parse(self, path: Path) -> ParsedDocument:
        text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")
        blocks: List[Block] = []
        page = 1
        paragraph: List[str] = []

        def flush():
            if paragraph:
                joined = normalize_space(" ".join(paragraph))
                if joined:
                    blocks.append(Block(BlockKind.PARAGRAPH, joined, page=page))
                paragraph.clear()

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if "\f" in line:                       # form feed = page break
                flush()
                page += line.count("\f")
                line = line.replace("\f", "")
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not line.strip():
                flush()
            elif _UNDERLINE.match(line) and paragraph is not None and not paragraph:
                pass                                # an underline we already consumed with its heading
            else:
                level = _heading_level(line, nxt) if not paragraph else 0
                if level:
                    flush()
                    blocks.append(Block(BlockKind.HEADING, normalize_space(line), level=level, page=page))
                    if nxt and _UNDERLINE.match(nxt):
                        i += 1
                elif _BULLET.match(line):
                    flush()
                    blocks.append(Block(BlockKind.LIST_ITEM, normalize_space(_BULLET.match(line).group(2)), page=page))
                else:
                    paragraph.append(line.strip())
            i += 1
        flush()
        title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), path.stem)
        return ParsedDocument(title=title, blocks=blocks, format="txt", page_count=page)
