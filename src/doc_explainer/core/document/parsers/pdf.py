"""PDF via PyMuPDF: layout-aware blocks with heading detection, tables, figures and captions.

Headings come from the PDF's outline when it has one, else from font size and weight relative to
the body text. Page headers/footers and bare page numbers are removed; paragraphs that continue
across a column or page break are rejoined.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf

from ...common.structure import Block, BlockKind, ParsedDocument
from .. import ocr
from ..sentences import normalize_space
from .markdown import table_markdown

_BULLET = re.compile(r"^\s*([•▪◦●■\-–—*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+(\S.*)$")
_CAPTION = re.compile(r"^(figure|fig\.?|table|chart|diagram|listing)\s*\d+[a-z]?\s*[.:\-–—]", re.I)
_TERMINAL = (".", "!", "?", ":", ";", '"', "”", "'", ")", "]")
_KEY_DIGITS = re.compile(r"\d+")
_NUMBERING = re.compile(r"^(?:(?:chapter|section|part|appendix)\s+)?(?:\d+(?:\.\d+)*|[ivxlc]+)[.):]?\s+", re.I)


def _norm_key(text: str) -> str:
    """For spotting repeated headers/footers: digits collapse so "Page 3" and "Page 4" match."""
    return _KEY_DIGITS.sub("#", normalize_space(text).lower())


def _title_key(text: str) -> str:
    """For matching a heading to its outline entry: ignore case, spacing and leading numbering."""
    return _NUMBERING.sub("", normalize_space(text)).lower().rstrip(".:")


class _Line:
    __slots__ = ("text", "size", "bold", "x0", "y0", "x1", "y1")

    def __init__(self, text, size, bold, bbox):
        self.text, self.size, self.bold = text, size, bold
        self.x0, self.y0, self.x1, self.y1 = bbox


class _Blk:
    __slots__ = ("lines", "page", "bbox")

    def __init__(self, lines, page, bbox):
        self.lines, self.page, self.bbox = lines, page, bbox

    @property
    def text(self) -> str:
        out = ""
        for ln in self.lines:
            t = ln.text.strip()
            if not t:
                continue
            if out.endswith("-") and t[:1].islower() and not out.endswith(" -"):
                out = out[:-1] + t
            else:
                out = (out + " " + t) if out else t
        return normalize_space(out)

    @property
    def size(self) -> float:
        total = sum(len(l.text) for l in self.lines) or 1
        return sum(l.size * len(l.text) for l in self.lines) / total

    @property
    def bold(self) -> bool:
        total = sum(len(l.text) for l in self.lines) or 1
        return sum(len(l.text) for l in self.lines if l.bold) / total > 0.7


class PdfParser:
    extensions = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        doc = pymupdf.open(str(path))
        try:
            pages = [self._page(doc[i], i + 1) for i in range(len(doc))]
            toc = doc.get_toc(simple=True) or []
            meta_title = (doc.metadata or {}).get("title", "") or ""
            page_count = len(doc)
        finally:
            doc.close()

        body = self._body_size([b for p in pages for b in p["text"]])
        self._drop_running_text(pages)
        heading_levels = self._heading_levels(pages, body, toc)

        blocks: List[Block] = []
        pending: Optional[Block] = None   # a paragraph that may continue on the next block/page

        def emit_pending():
            nonlocal pending
            if pending is not None:
                blocks.append(pending)
                pending = None

        for p in pages:
            items: List[Tuple[float, float, str, object]] = []
            for b in p["text"]:
                items.append((b.bbox[1], b.bbox[0], "text", b))
            for t in p["tables"]:
                items.append((t["bbox"][1], t["bbox"][0], "table", t))
            for im in p["images"]:
                items.append((im["bbox"][1], im["bbox"][0], "image", im))
            for _, _, kind, obj in sorted(items, key=lambda x: (round(x[0], 0), x[1])):
                if kind == "table":
                    emit_pending()
                    blocks.append(Block(BlockKind.TABLE, table_markdown(obj["rows"]), page=p["no"], attrs={"rows": obj["rows"]}))
                elif kind == "image":
                    emit_pending()
                    blocks.append(Block(BlockKind.IMAGE, "", page=p["no"], attrs={"width": obj["w"], "height": obj["h"]}))
                else:
                    pending = self._text_block(obj, p["no"], heading_levels, blocks, pending, emit_pending)
        emit_pending()

        title = normalize_space(meta_title) if meta_title and len(meta_title) > 3 else ""
        if not title:
            title = next((b.text for b in blocks if b.kind is BlockKind.HEADING), Path(path).stem)
        return ParsedDocument(title=title, blocks=blocks, format="pdf", page_count=max(page_count, 1))

    # ------------------------------------------------------------------ one page
    def _page(self, page, number: int) -> Dict:
        raw = page.get_text("dict", sort=True)
        tables, table_boxes = [], []
        try:
            for t in page.find_tables().tables:
                rows = [[normalize_space(c or "") for c in r] for r in t.extract()]
                rows = [r for r in rows if any(r)]
                filled = sum(1 for r in rows for c in r if c)
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2 and filled >= 3:
                    tables.append({"bbox": tuple(t.bbox), "rows": rows})
                    table_boxes.append(tuple(t.bbox))
        except Exception:
            pass
        blocks: List[_Blk] = []
        for b in raw["blocks"]:
            if b["type"] != 0:
                continue
            if any(self._overlap(tuple(b["bbox"]), tb) > 0.6 for tb in table_boxes):
                continue
            lines = []
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                chars = sum(len(s["text"]) for s in spans)
                size = sum(s["size"] * len(s["text"]) for s in spans) / chars
                bold = sum(len(s["text"]) for s in spans if (s["flags"] & 16) or "bold" in s["font"].lower()) / chars > 0.6
                lines.append(_Line("".join(s["text"] for s in spans), size, bold, ln["bbox"]))
            if lines:
                blocks.append(_Blk(lines, number, tuple(b["bbox"])))
        if not blocks and not tables and page.get_images():      # a scanned page: the text is only in the picture
            for paragraph in ocr.group_paragraphs(ocr.ocr_page(page)):
                lines = [_Line(text, max(bbox[3] - bbox[1], 1.0), False, bbox) for bbox, text in paragraph]
                blocks.append(_Blk(lines, number, (min(l.x0 for l in lines), min(l.y0 for l in lines),
                                                   max(l.x1 for l in lines), max(l.y1 for l in lines))))
        images = []
        try:
            area = page.rect.width * page.rect.height
            for info in page.get_image_info():
                bb = info["bbox"]
                if (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0.03 * area:
                    images.append({"bbox": tuple(bb), "w": info.get("width", 0), "h": info.get("height", 0)})
        except Exception:
            pass
        return {"no": number, "text": blocks, "tables": tables, "images": images, "height": page.rect.height}

    @staticmethod
    def _overlap(a, b) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        return max(w, 0) * max(h, 0) / area

    # ------------------------------------------------------------------ document-wide analysis
    @staticmethod
    def _body_size(blocks: List[_Blk]) -> float:
        counts: Counter = Counter()
        for b in blocks:
            for ln in b.lines:
                counts[round(ln.size * 2) / 2] += len(ln.text)
        return counts.most_common(1)[0][0] if counts else 11.0

    @staticmethod
    def _drop_running_text(pages: List[Dict]) -> None:
        """Remove page numbers and headers/footers that repeat on many pages."""
        many = len(pages) >= 3
        seen: Counter = Counter()
        for p in pages:
            keys = set()
            for b in p["text"]:
                if b.bbox[1] < 0.1 * p["height"] or b.bbox[3] > 0.9 * p["height"]:
                    keys.add(_norm_key(b.text))
            seen.update(keys)
        for p in pages:
            keep = []
            for b in p["text"]:
                edge = b.bbox[1] < 0.1 * p["height"] or b.bbox[3] > 0.9 * p["height"]
                text = b.text
                if edge and (re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", text.lower())
                             or (many and seen[_norm_key(text)] >= max(3, 0.4 * len(pages)))):
                    continue
                keep.append(b)
            p["text"] = keep

    def _heading_levels(self, pages, body: float, toc) -> Dict[int, int]:
        """id(block) -> heading level, from the outline when it matches, else from font size."""
        by_title: Dict[Tuple[int, str], int] = {}
        for level, title, page in toc:
            by_title.setdefault((page, _title_key(title)), level)
        levels: Dict[int, int] = {}
        candidates: List[_Blk] = []
        for p in pages:
            for b in p["text"]:
                text = b.text
                if not text or len(text.split()) > 20 or len(b.lines) > 3 or _CAPTION.match(text):
                    continue
                key = (p["no"], _title_key(text))
                if key in by_title:
                    levels[id(b)] = by_title[key]
                elif (b.size >= body * 1.12 or (b.bold and b.size >= body * 0.98 and len(text.split()) <= 12)) \
                        and not text.rstrip().endswith((".", ",", ";")) and not _BULLET.match(text):
                    candidates.append(b)
        if candidates:
            sizes = sorted({round(b.size) for b in candidates}, reverse=True)
            for b in candidates:
                if id(b) not in levels:
                    rank = sizes.index(round(b.size)) + 1
                    levels[id(b)] = rank
        return levels

    # ------------------------------------------------------------------ one text block -> blocks
    def _text_block(self, b: _Blk, page: int, levels, out: List[Block], pending: Optional[Block], emit_pending):
        text = b.text
        if not text:
            return pending
        if id(b) in levels:
            emit_pending()
            out.append(Block(BlockKind.HEADING, text, level=levels[id(b)], page=page))
            return None
        if _CAPTION.match(text):
            emit_pending()
            out.append(Block(BlockKind.CAPTION, text, page=page))
            return None
        if _BULLET.match(b.lines[0].text):
            emit_pending()
            items: List[str] = []
            for ln in b.lines:
                if _BULLET.match(ln.text) or not items:
                    items.append(ln.text.strip())
                else:
                    items[-1] += " " + ln.text.strip()
            for it in items:
                m = _BULLET.match(it)
                out.append(Block(BlockKind.LIST_ITEM, normalize_space(m.group(2) if m else it), page=page))
            return None
        # a paragraph; join with the previous one if that one stopped mid-sentence
        if pending is not None and not pending.text.endswith(_TERMINAL) and text[:1].islower():
            return Block(BlockKind.PARAGRAPH, normalize_space(pending.text + " " + text), page=pending.page)
        emit_pending()
        return Block(BlockKind.PARAGRAPH, text, page=page)
