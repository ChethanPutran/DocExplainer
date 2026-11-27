"""PowerPoint: each slide is a section (its title), its text frames are paragraphs and list items."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from ...common.structure import Block, BlockKind, ParsedDocument
from ..sentences import normalize_space
from .markdown import table_markdown


class PptxParser:
    extensions = (".pptx",)

    def parse(self, path: Path) -> ParsedDocument:
        prs = Presentation(str(path))
        blocks: List[Block] = []
        for number, slide in enumerate(prs.slides, 1):
            title_shape = slide.shapes.title
            title = normalize_space(title_shape.text_frame.text) if title_shape is not None and title_shape.has_text_frame else ""
            blocks.append(Block(BlockKind.HEADING, title or f"Slide {number}", level=1, page=number))
            for shape in slide.shapes:
                if shape is title_shape:
                    continue
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    alt = normalize_space(shape._element.xpath("string(.//p:cNvPr/@descr)") or shape.name)
                    blocks.append(Block(BlockKind.IMAGE, alt, page=number))
                elif getattr(shape, "has_table", False) and shape.has_table:
                    rows = [[normalize_space(c.text) for c in r.cells] for r in shape.table.rows]
                    rows = [r for r in rows if any(r)]
                    if rows:
                        blocks.append(Block(BlockKind.TABLE, table_markdown(rows), page=number, attrs={"rows": rows}))
                elif shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = normalize_space("".join(r.text for r in para.runs))
                        if text:
                            kind = BlockKind.LIST_ITEM if para.level > 0 or len(shape.text_frame.paragraphs) > 1 else BlockKind.PARAGRAPH
                            blocks.append(Block(kind, text, page=number))
            if slide.has_notes_slide and normalize_space(slide.notes_slide.notes_text_frame.text):
                blocks.append(Block(BlockKind.PARAGRAPH, normalize_space(slide.notes_slide.notes_text_frame.text),
                                    page=number, attrs={"notes": True}))
        title = (prs.core_properties.title or "").strip() or (blocks[0].text if blocks else path.stem)
        return ParsedDocument(title=title, blocks=blocks, format="pptx", page_count=max(len(prs.slides), 1))
