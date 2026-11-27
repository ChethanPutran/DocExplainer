"""The unified document structure: what every parser produces and every later stage consumes.

Parsers emit a flat, ordered stream of ``Block``s (a format-independent vocabulary). One
``StructureBuilder`` turns that into a ``DocumentStructure``: a tree of document > section >
paragraph > sentence (plus tables, figures, equations) with explicit relationships between parts.

Everything has an ``ordinal``: its position in reading order. The ordinal is the "time axis" of the
whole system: the knowledge graph grows along it, the learner's position is one, and retrieval can
be limited to ``ordinal <= position`` so answers use only what has been read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class BlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    EQUATION = "equation"
    CODE = "code"


@dataclass(frozen=True)
class Block:
    """One piece of content as a parser saw it. ``level`` is for headings (1 = top)."""
    kind: BlockKind
    text: str = ""
    level: int = 0
    page: int = 1
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    title: str
    blocks: List[Block]
    format: str
    page_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeKind(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CODE = "code"


class EdgeKind(str, Enum):
    CONTAINS = "contains"      # parent -> child
    NEXT = "next"              # a node -> its next sibling in reading order
    REFERS_TO = "refers_to"    # a paragraph -> the figure/table/section it mentions
    CAPTIONS = "captions"      # a caption paragraph -> the figure/table it describes


@dataclass
class Node:
    id: str
    kind: NodeKind
    parent_id: Optional[str]
    ordinal: int               # reading order, unique within the document, starts at 0
    depth: int
    page_start: int
    page_end: int
    title: str = ""            # sections: the heading text
    text: str = ""             # leaf content (paragraph, sentence, table as markdown, figure alt/caption)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: EdgeKind


class DocumentStructure:
    """An immutable-in-spirit view over the nodes and edges, with the queries the system needs."""

    def __init__(self, doc_id: str, title: str, source_format: str, page_count: int,
                 nodes: List[Node], edges: List[Edge]):
        self.doc_id, self.title, self.source_format, self.page_count = doc_id, title, source_format, page_count
        self.nodes = sorted(nodes, key=lambda n: n.ordinal)
        self.edges = list(edges)
        self._by_id = {n.id: n for n in self.nodes}
        self._children: Dict[str, List[Node]] = {}
        for n in self.nodes:
            if n.parent_id is not None:
                self._children.setdefault(n.parent_id, []).append(n)

    # ------------------------------------------------------------------ lookups
    @property
    def root(self) -> Node:
        return self.nodes[0]

    def node(self, node_id: str) -> Node:
        return self._by_id[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._by_id

    def children(self, node_id: str) -> List[Node]:
        return self._children.get(node_id, [])

    def sections(self) -> List[Node]:
        """All sections in reading order."""
        return [n for n in self.nodes if n.kind is NodeKind.SECTION]

    def section_by_ordinal(self, ordinal: int) -> Optional[Node]:
        n = next((x for x in self.nodes if x.ordinal == ordinal), None)
        return n if n is not None and n.kind is NodeKind.SECTION else None

    def edges_of(self, kind: EdgeKind) -> List[Edge]:
        return [e for e in self.edges if e.kind is kind]

    # ------------------------------------------------------------------ structure queries
    def section_of(self, node_id: str) -> Optional[Node]:
        n = self._by_id[node_id]
        while n is not None and n.kind is not NodeKind.SECTION:
            n = self._by_id.get(n.parent_id) if n.parent_id else None
        return n

    def section_path(self, node_id: str) -> List[str]:
        """Titles from the top-level section down to this node's section."""
        titles: List[str] = []
        n: Optional[Node] = self._by_id[node_id]
        while n is not None:
            if n.kind is NodeKind.SECTION and n.title:
                titles.append(n.title)
            n = self._by_id.get(n.parent_id) if n.parent_id else None
        return list(reversed(titles))

    def own_content(self, section_id: str) -> List[Node]:
        """The section's direct content (paragraphs, tables, figures, ...), not its subsections."""
        return [c for c in self.children(section_id) if c.kind is not NodeKind.SECTION]

    def own_end(self, section_id: str) -> int:
        """Last ordinal of the section's own content: the reader's position after finishing it."""
        end = self._by_id[section_id].ordinal
        for c in self.own_content(section_id):
            end = max(end, self.subtree_end(c.id))
        return end

    def subtree_end(self, node_id: str) -> int:
        n = self._by_id[node_id]
        return max([n.ordinal] + [d.ordinal for d in self._descendants(n)])

    def _descendants(self, node: Node) -> Iterator[Node]:
        stack = list(self.children(node.id))
        while stack:
            c = stack.pop()
            yield c
            stack.extend(self.children(c.id))

    def section_at_page(self, page: int) -> Optional[Node]:
        """The deepest, latest section whose own content is on ``page`` (else the last one before it)."""
        best, best_key = None, None
        for s in self.sections():
            own = self.own_content(s.id)
            lo = min([s.page_start] + [c.page_start for c in own])
            hi = max([s.page_start] + [c.page_end for c in own])
            if lo <= page <= hi:
                key = (s.depth, s.ordinal)
                if best_key is None or key > best_key:
                    best, best_key = s, key
        if best is not None:
            return best
        earlier = [s for s in self.sections() if s.page_start <= page]
        return earlier[-1] if earlier else (self.sections()[0] if self.sections() else None)

    def text_of(self, section_id: str, include_subsections: bool = False) -> str:
        """The section's paragraph text in reading order (tables and figures as their text)."""
        parts: List[str] = []
        for c in self.children(section_id):
            if c.kind is NodeKind.SECTION:
                if include_subsections:
                    parts.append(self.text_of(c.id, True))
            elif c.kind is not NodeKind.SENTENCE and c.text:
                parts.append(c.text)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title, "source_format": self.source_format,
            "page_count": self.page_count,
            "nodes": [{"id": n.id, "kind": n.kind.value, "parent_id": n.parent_id, "ordinal": n.ordinal,
                       "depth": n.depth, "page_start": n.page_start, "page_end": n.page_end,
                       "title": n.title, "text": n.text, "attrs": n.attrs} for n in self.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind.value} for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentStructure":
        nodes = [Node(id=n["id"], kind=NodeKind(n["kind"]), parent_id=n["parent_id"], ordinal=n["ordinal"],
                      depth=n["depth"], page_start=n["page_start"], page_end=n["page_end"],
                      title=n.get("title", ""), text=n.get("text", ""), attrs=n.get("attrs", {}))
                 for n in d["nodes"]]
        edges = [Edge(e["src"], e["dst"], EdgeKind(e["kind"])) for e in d["edges"]]
        return cls(d["doc_id"], d["title"], d["source_format"], d["page_count"], nodes, edges)

    def without_text(self) -> "DocumentStructure":
        """The same structure with body text removed (section titles stay: they are structural labels)."""
        nodes = [Node(n.id, n.kind, n.parent_id, n.ordinal, n.depth, n.page_start, n.page_end,
                      n.title, "", {k: v for k, v in n.attrs.items() if k in ("label", "caption_for", "list", "depth")})
                 for n in self.nodes]
        return DocumentStructure(self.doc_id, self.title, self.source_format, self.page_count, nodes, self.edges)
