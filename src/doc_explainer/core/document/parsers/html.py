"""HTML: headings h1-h6, paragraphs, lists, tables, figures and images, in document order."""
from __future__ import annotations

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, NavigableString, Tag

from ...common.structure import Block, BlockKind, ParsedDocument
from ..sentences import normalize_space
from .markdown import table_markdown
from .text import read_text

_SKIP = {"script", "style", "nav", "footer", "aside", "noscript", "form", "header", "svg", "template"}
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_TEXT_BLOCKS = {"p", "blockquote", "dd", "dt", "summary", "address"}
_CONTAINERS = {"div", "section", "article", "main", "body", "span", "ul", "ol", "dl", "details", "li", "figure"}


def _rows(table: Tag) -> List[List[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [normalize_space(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return rows


class HtmlParser:
    extensions = (".html", ".htm", ".xhtml")

    def parse(self, path: Path) -> ParsedDocument:
        soup = BeautifulSoup(read_text(path), "lxml")
        title = normalize_space(soup.title.get_text()) if soup.title and soup.title.get_text() else ""
        blocks: List[Block] = []
        root = soup.body or soup
        self._walk(root, blocks)
        if not title:
            title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), path.stem)
        return ParsedDocument(title=title, blocks=blocks, format="html")

    def _walk(self, node: Tag, out: List[Block]) -> None:
        loose: List[str] = []

        def flush():
            text = normalize_space(" ".join(loose))
            loose.clear()
            if text:
                out.append(Block(BlockKind.PARAGRAPH, text))

        for child in node.children:
            if isinstance(child, NavigableString):
                if child.strip() and type(child).__name__ not in ("Comment", "Doctype", "Declaration"):
                    loose.append(str(child))
                continue
            name = child.name
            if name in _SKIP:
                continue
            if name in _HEADINGS:
                flush()
                text = normalize_space(child.get_text(" "))
                if text:
                    out.append(Block(BlockKind.HEADING, text, level=_HEADINGS[name]))
            elif name in _TEXT_BLOCKS:
                flush()
                text = normalize_space(child.get_text(" "))
                if text:
                    out.append(Block(BlockKind.PARAGRAPH, text))
            elif name == "li":
                flush()
                nested = child.find(["ul", "ol"])
                own = normalize_space(" ".join(s for s in child.find_all(string=True, recursive=False)))
                text = own or normalize_space(child.get_text(" "))
                if text:
                    out.append(Block(BlockKind.LIST_ITEM, text))
                if nested:
                    self._walk(nested, out)
            elif name == "table":
                flush()
                rows = _rows(child)
                if rows:
                    out.append(Block(BlockKind.TABLE, table_markdown(rows), attrs={"rows": rows}))
                cap = child.find("caption")
                if cap and normalize_space(cap.get_text()):
                    out.insert(len(out) - 1, Block(BlockKind.CAPTION, normalize_space(cap.get_text(" "))))
            elif name == "figure":
                flush()
                for img in child.find_all("img"):
                    out.append(Block(BlockKind.IMAGE, normalize_space(img.get("alt", "")), attrs={"src": img.get("src", "")}))
                cap = child.find("figcaption")
                if cap and normalize_space(cap.get_text()):
                    out.append(Block(BlockKind.CAPTION, normalize_space(cap.get_text(" "))))
            elif name == "img":
                flush()
                out.append(Block(BlockKind.IMAGE, normalize_space(child.get("alt", "")), attrs={"src": child.get("src", "")}))
            elif name == "pre":
                flush()
                code = child.get_text()
                if code.strip():
                    out.append(Block(BlockKind.CODE, code.rstrip("\n")))
            elif name == "br":
                loose.append(" ")
            elif name in _CONTAINERS or isinstance(child, Tag):
                if name in ("ul", "ol", "dl", "div", "section", "article", "main", "details", "span", "body"):
                    if name == "span" and not child.find(list(_HEADINGS) + ["p", "table", "ul", "ol", "img"]):
                        loose.append(child.get_text(" "))
                        continue
                    flush()
                    self._walk(child, out)
                else:
                    loose.append(child.get_text(" "))   # inline: a, b, i, em, strong, code, ...
        flush()
