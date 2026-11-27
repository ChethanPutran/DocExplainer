"""Word documents: heading styles, lists, tables and inline images, in body order."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from ...common.structure import Block, BlockKind, ParsedDocument
from ..sentences import normalize_space
from .markdown import table_markdown

_HEADING = re.compile(r"^heading\s*(\d+)$", re.I)
_NS_DRAWING = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
_NS_BR = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
_NS_PAGE_RENDERED = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}lastRenderedPageBreak"


def _page_breaks(p: Paragraph) -> int:
    n = len(p._p.findall(f".//{_NS_PAGE_RENDERED}"))
    n += sum(1 for br in p._p.iter(_NS_BR) if br.get(f"{_NS_BR[:-2]}type") == "page")
    return n


class DocxParser:
    extensions = (".docx",)

    def parse(self, path: Path) -> ParsedDocument:
        doc = DocxDocument(str(path))
        blocks: List[Block] = []
        page = 1
        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(child, doc)
                text = normalize_space(p.text)
                style = (p.style.name if p.style is not None else "") or ""
                m = _HEADING.match(style)
                if style.lower() == "title" and text:
                    blocks.append(Block(BlockKind.HEADING, text, level=1, page=page))
                elif m and text:
                    blocks.append(Block(BlockKind.HEADING, text, level=int(m.group(1)), page=page))
                elif style.lower().startswith("caption") and text:
                    blocks.append(Block(BlockKind.CAPTION, text, page=page))
                elif style.lower().startswith("list") and text:
                    blocks.append(Block(BlockKind.LIST_ITEM, text, page=page))
                elif text:
                    blocks.append(Block(BlockKind.PARAGRAPH, text, page=page))
                for drawing in p._p.iter(_NS_DRAWING):
                    props = next((e for e in drawing.iter() if e.tag.endswith("}docPr")), None)
                    alt = normalize_space((props.get("descr") or props.get("name") or "") if props is not None else "")
                    blocks.append(Block(BlockKind.IMAGE, alt, page=page))
                page += _page_breaks(p)
            elif tag == "tbl":
                t = Table(child, doc)
                rows = [[normalize_space(c.text) for c in r.cells] for r in t.rows]
                rows = [r for r in rows if any(r)]
                if rows:
                    blocks.append(Block(BlockKind.TABLE, table_markdown(rows), page=page, attrs={"rows": rows}))
        title = (doc.core_properties.title or "").strip() or next(
            (b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), path.stem)
        return ParsedDocument(title=title, blocks=blocks, format="docx", page_count=page)
