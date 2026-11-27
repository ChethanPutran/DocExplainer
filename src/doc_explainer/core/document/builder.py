"""StructureBuilder: the one place that turns a parser's flat block stream into a DocumentStructure.

Format-independent by construction: every parser feeds the same builder, so a PDF, a web page and
a Word file with the same content end up with the same shape.

  document > section (nested by heading level) > paragraph > sentence
                                              > table | figure | equation | code

with CONTAINS and NEXT edges everywhere, REFERS_TO edges from paragraphs that mention a figure,
table or section ("see Table 2"), and CAPTIONS edges from captions to what they describe.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..common.structure import (Block, BlockKind, DocumentStructure, Edge, EdgeKind, Node, NodeKind,
                                ParsedDocument)
from .sentences import normalize_space, split_sentences

_CAPTION_LABEL = re.compile(r"^(figure|fig\.?|table|chart|diagram|listing)\s*(\d+[a-z]?)", re.I)
_CAPTION_LINE = re.compile(r"^(figure|fig\.?|table|chart|diagram|listing)\s*\d+[a-z]?\s*[.:\-–—]\s*\S", re.I)
_SECTION_NUMBER = re.compile(r"^(?:(chapter|section|part|appendix)\s+)?(\d+(?:\.\d+)*|[IVXLC]+)[.):]?\s+\S", re.I)
_CHAPTER_ONLY = re.compile(r"^(chapter|section|part|appendix)\s+(\d+|[IVXLC]+)\b", re.I)
_REFERENCE = re.compile(r"\b(figure|fig\.?|table|chart|diagram|listing|section|sec\.?|chapter|appendix)\s*(\d+(?:\.\d+)*[a-z]?)", re.I)
_KIND_ALIASES = {"fig": "figure", "fig.": "figure", "sec": "section", "sec.": "section", "chart": "figure", "diagram": "figure"}


def _label(kind: str, number: str) -> str:
    kind = kind.lower()
    return f"{_KIND_ALIASES.get(kind, kind)} {number.lower()}"


class StructureBuilder:
    """``fallback_window_words``: documents with no headings at all are cut into synthetic sections of
    about this many words, so the rest of the system always has sections to work with."""

    def __init__(self, fallback_window_words: int = 900):
        self.window = fallback_window_words

    def build(self, parsed: ParsedDocument, doc_id: str) -> DocumentStructure:
        blocks = self._clean(parsed.blocks)
        blocks = self._normalise_levels(blocks)
        if not any(b.kind is BlockKind.HEADING for b in blocks):
            blocks = self._synthetic_sections(blocks)
        nodes, edges = self._tree(blocks, parsed, doc_id)
        self._captions(nodes, edges)
        self._references(nodes, edges)
        title = parsed.title or next((n.title for n in nodes if n.kind is NodeKind.SECTION), doc_id)
        return DocumentStructure(doc_id, title, parsed.format, parsed.page_count, nodes, edges)

    # ------------------------------------------------------------------ block preparation
    @staticmethod
    def _clean(blocks: List[Block]) -> List[Block]:
        out = []
        for b in blocks:
            text = b.text if b.kind is BlockKind.CODE else normalize_space(b.text)
            kind = b.kind
            # whichever parser it came from, a short "Figure 3: ..." / "Table 2. ..." line is a caption
            if kind is BlockKind.PARAGRAPH and len(text) <= 300 and _CAPTION_LINE.match(text):
                kind = BlockKind.CAPTION
            if text or kind in (BlockKind.TABLE, BlockKind.IMAGE):
                out.append(Block(kind, text, b.level, max(b.page, 1), b.attrs))
        return out

    @staticmethod
    def _normalise_levels(blocks: List[Block]) -> List[Block]:
        """Make the shallowest heading level 1 and never skip a level going down."""
        levels = [b.level for b in blocks if b.kind is BlockKind.HEADING]
        if not levels:
            return blocks
        shift = min(levels) - 1
        out, prev = [], 0
        for b in blocks:
            if b.kind is BlockKind.HEADING:
                level = min(max(b.level - shift, 1), prev + 1)
                prev = level
                b = Block(b.kind, b.text, level, b.page, b.attrs)
            out.append(b)
        return out

    def _synthetic_sections(self, blocks: List[Block]) -> List[Block]:
        out, words, part = [], 0, 0
        for b in blocks:
            if words == 0 or (words >= self.window and b.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM)):
                part += 1
                out.append(Block(BlockKind.HEADING, f"Part {part}", level=1, page=b.page, attrs={"synthetic": True}))
                words = 0
            out.append(b)
            words += len(b.text.split())
        return out

    # ------------------------------------------------------------------ the tree
    def _tree(self, blocks: List[Block], parsed: ParsedDocument, doc_id: str) -> Tuple[List[Node], List[Edge]]:
        nodes: List[Node] = []
        edges: List[Edge] = []
        counter = 0

        def add(kind, node_id, parent: Node, page, title="", text="", attrs=None) -> Node:
            nonlocal counter
            counter += 1
            n = Node(node_id, kind, parent.id, counter, parent.depth + 1, page, page, title, text, attrs or {})
            nodes.append(n)
            edges.append(Edge(parent.id, n.id, EdgeKind.CONTAINS))
            return n

        root = Node("doc", NodeKind.DOCUMENT, None, 0, 0, 1, max(parsed.page_count, 1), parsed.title, "", {})
        nodes.append(root)
        stack: List[Tuple[int, Node]] = [(0, root)]
        child_counts: Dict[Tuple[str, str], int] = {}

        def next_id(parent: Node, prefix: str) -> str:
            key = (parent.id, prefix)
            child_counts[key] = child_counts.get(key, 0) + 1
            n = child_counts[key]
            if prefix == "s":
                return f"s{n}" if parent.kind is NodeKind.DOCUMENT else f"{parent.id}.{n}"
            return f"{parent.id}/{prefix}{n}"

        def current_section(page: int) -> Node:
            if len(stack) == 1:        # content before any heading
                sec = add(NodeKind.SECTION, next_id(root, "s"), root, page, title="Preamble", attrs={"synthetic": True})
                stack.append((1, sec))
            return stack[-1][1]

        for b in blocks:
            if b.kind is BlockKind.HEADING:
                while len(stack) > 1 and stack[-1][0] >= b.level:
                    stack.pop()
                parent = stack[-1][1]
                sec = add(NodeKind.SECTION, next_id(parent, "s"), parent, b.page, title=b.text,
                          attrs={"synthetic": True} if b.attrs.get("synthetic") else {})
                stack.append((b.level, sec))
                continue
            sec = current_section(b.page)
            if b.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.CAPTION):
                attrs = {}
                if b.kind is BlockKind.LIST_ITEM:
                    attrs["list"] = True
                if b.kind is BlockKind.CAPTION:
                    attrs["caption"] = True
                p = add(NodeKind.PARAGRAPH, next_id(sec, "p"), sec, b.page, text=b.text, attrs=attrs)
                if b.kind is not BlockKind.CAPTION:
                    for sentence in split_sentences(b.text):
                        add(NodeKind.SENTENCE, next_id(p, "n"), p, b.page, text=sentence)
            elif b.kind is BlockKind.TABLE:
                add(NodeKind.TABLE, next_id(sec, "t"), sec, b.page, text=b.text, attrs={"rows": len(b.attrs.get("rows", []))})
            elif b.kind is BlockKind.IMAGE:
                add(NodeKind.FIGURE, next_id(sec, "f"), sec, b.page, text=b.text, attrs={k: v for k, v in b.attrs.items() if k in ("src", "width", "height")})
            elif b.kind is BlockKind.EQUATION:
                add(NodeKind.EQUATION, next_id(sec, "e"), sec, b.page, text=b.text)
            elif b.kind is BlockKind.CODE:
                add(NodeKind.CODE, next_id(sec, "c"), sec, b.page, text=b.text)

        # page spans: a node reaches as far as anything inside it (it starts where it begins)
        by_id = {n.id: n for n in nodes}
        for n in reversed(nodes):
            if n.parent_id:
                parent = by_id[n.parent_id]
                parent.page_end = max(parent.page_end, n.page_end)
        # NEXT: consecutive siblings, in reading order
        siblings: Dict[str, List[Node]] = {}
        for n in nodes:
            if n.parent_id:
                siblings.setdefault(n.parent_id, []).append(n)
        for group in siblings.values():
            for a, b in zip(group, group[1:]):
                edges.append(Edge(a.id, b.id, EdgeKind.NEXT))
        return nodes, edges

    # ------------------------------------------------------------------ relationships
    @staticmethod
    def _captions(nodes: List[Node], edges: List[Edge]) -> None:
        """Link each caption to the figure/table it describes, and give that figure/table its label."""
        children: Dict[str, List[Node]] = {}
        for n in nodes:
            if n.parent_id:
                children.setdefault(n.parent_id, []).append(n)
        for group in children.values():
            for i, n in enumerate(group):
                if not (n.kind is NodeKind.PARAGRAPH and n.attrs.get("caption")):
                    continue
                m = _CAPTION_LABEL.match(n.text)
                want = NodeKind.TABLE if m and m.group(1).lower() == "table" else NodeKind.FIGURE
                # tables are usually captioned above, figures below: look that way first
                order = (1, -1) if want is NodeKind.TABLE else (-1, 1)
                target = None
                for step in order:
                    j = i + step
                    if 0 <= j < len(group) and group[j].kind is want and "caption_for" not in group[j].attrs \
                            and "label" not in group[j].attrs:
                        target = group[j]
                        break
                if target is None:
                    continue
                edges.append(Edge(n.id, target.id, EdgeKind.CAPTIONS))
                target.attrs["caption"] = n.text
                if m:
                    target.attrs["label"] = _label(m.group(1), m.group(2))
                    n.attrs["caption_for"] = target.id
                if not target.text:
                    target.text = n.text

    @staticmethod
    def _references(nodes: List[Node], edges: List[Edge]) -> None:
        """REFERS_TO edges from paragraphs to the figures, tables and sections they mention."""
        targets: Dict[str, Node] = {}
        for n in nodes:
            if n.kind in (NodeKind.FIGURE, NodeKind.TABLE) and n.attrs.get("label"):
                targets.setdefault(n.attrs["label"], n)
            elif n.kind is NodeKind.SECTION:
                m = _CHAPTER_ONLY.match(n.title)
                if m:
                    label = f"{m.group(1).lower()} {m.group(2).lower()}"
                else:
                    m = _SECTION_NUMBER.match(n.title)
                    label = f"{(m.group(1) or 'section').lower()} {m.group(2).lower()}" if m else None
                if label:
                    n.attrs["label"] = label
                    targets.setdefault(label, n)
        if not targets:
            return
        seen = set()
        for n in nodes:
            if n.kind is not NodeKind.PARAGRAPH or n.attrs.get("caption"):
                continue
            for m in _REFERENCE.finditer(n.text):
                target = targets.get(_label(m.group(1), m.group(2)))
                if target is not None and target.id != n.parent_id and (n.id, target.id) not in seen:
                    seen.add((n.id, target.id))
                    edges.append(Edge(n.id, target.id, EdgeKind.REFERS_TO))
