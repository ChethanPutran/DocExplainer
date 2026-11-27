"""Markdown: ATX and setext headings, fenced code, lists, pipe tables, images, paragraphs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ...common.structure import Block, BlockKind, ParsedDocument
from ..sentences import normalize_space
from .text import read_text

_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)\s]+)[^)]*\)\s*$")
_INLINE = [(re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"), (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
           (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"), (re.compile(r"(?<!\w)([*_])(.+?)\1(?!\w)"), r"\2"),
           (re.compile(r"`([^`]*)`"), r"\1")]


def _plain(s: str) -> str:
    for rx, rep in _INLINE:
        s = rx.sub(rep, s)
    return normalize_space(s)


def _cells(line: str) -> List[str]:
    return [_plain(c) for c in line.strip().strip("|").split("|")]


def table_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(out)


class MarkdownParser:
    extensions = (".md", ".markdown", ".mdown")

    def parse(self, path: Path) -> ParsedDocument:
        text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")
        if text.startswith("---\n"):                       # front matter
            end = text.find("\n---", 4)
            if end != -1:
                text = text[end + 4:].lstrip("\n")
        lines = text.split("\n")
        blocks: List[Block] = []
        para: List[str] = []

        def flush():
            if para:
                blocks.append(Block(BlockKind.PARAGRAPH, _plain(" ".join(para))))
                para.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _FENCE.match(line):
                flush()
                code: List[str] = []
                i += 1
                while i < len(lines) and not _FENCE.match(lines[i]):
                    code.append(lines[i])
                    i += 1
                blocks.append(Block(BlockKind.CODE, "\n".join(code)))
            elif _ATX.match(line):
                flush()
                m = _ATX.match(line)
                blocks.append(Block(BlockKind.HEADING, _plain(m.group(2)), level=len(m.group(1))))
            elif line.strip() and not para and _SETEXT.match(nxt) and not _LIST.match(line) and not line.startswith("|"):
                blocks.append(Block(BlockKind.HEADING, _plain(line), level=1 if nxt.strip()[0] == "=" else 2))
                i += 1
            elif line.lstrip().startswith("|") and _TABLE_SEP.match(nxt):
                flush()
                rows = [_cells(line)]
                i += 2
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    rows.append(_cells(lines[i]))
                    i += 1
                i -= 1
                blocks.append(Block(BlockKind.TABLE, table_markdown(rows), attrs={"rows": rows}))
            elif _IMAGE.match(line):
                flush()
                m = _IMAGE.match(line)
                blocks.append(Block(BlockKind.IMAGE, normalize_space(m.group(1)), attrs={"src": m.group(2)}))
            elif _LIST.match(line):
                flush()
                blocks.append(Block(BlockKind.LIST_ITEM, _plain(_LIST.match(line).group(2))))
            elif not line.strip():
                flush()
            else:
                para.append(line.strip())
            i += 1
        flush()
        title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), path.stem)
        return ParsedDocument(title=title, blocks=blocks, format="markdown")
