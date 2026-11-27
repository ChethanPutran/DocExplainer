"""Turns a DocumentStructure into the pieces that get embedded.

A chunk is a run of whole paragraphs from ONE section (about a paragraph or three), so it never
straddles a heading and always has a precise place: its first node's reading-order ordinal.
Tables and figures are chunks of their own (with their caption); a paragraph too long for one chunk
is cut at sentence boundaries.
"""
from __future__ import annotations

from typing import List

from ..common.ports import Chunk
from ..common.structure import DocumentStructure, Node, NodeKind

MIN_CHARS = 250       # keep adding paragraphs until a chunk is at least this long ...
MAX_CHARS = 1000      # ... but never beyond this, unless a single sentence is longer
MAX_TABLE_CHARS = 3000


def _chunk(structure: DocumentStructure, section: Node, first: Node, text: str, suffix: str = "") -> Chunk:
    return Chunk(id=first.id + suffix, text=text, node_id=first.id, kind=first.kind.value, ordinal=first.ordinal,
                 page=first.page_start, section_id=section.id, section_path=" > ".join(structure.section_path(section.id)))


def _split_long(structure: DocumentStructure, section: Node, para: Node) -> List[Chunk]:
    sentences = [s for s in structure.children(para.id) if s.kind is NodeKind.SENTENCE] or []
    if not sentences:
        return [_chunk(structure, section, para, para.text[:MAX_CHARS * 2])]
    chunks, buf, first, part = [], "", sentences[0], 0
    for s in sentences:
        if buf and len(buf) + len(s.text) + 1 > MAX_CHARS:
            chunks.append(_chunk(structure, section, first, buf, f"~{part}" if part else ""))
            buf, first, part = "", s, part + 1
        buf = f"{buf} {s.text}".strip()
    if buf:
        chunks.append(_chunk(structure, section, first, buf, f"~{part}" if part else ""))
    return chunks


def chunk_structure(structure: DocumentStructure) -> List[Chunk]:
    """All chunks of the document, in reading order."""
    captions = {e.dst: e.src for e in structure.edges if e.kind.value == "captions"}      # figure/table -> caption paragraph
    caption_ids = set(captions.values())
    out: List[Chunk] = []
    for section in structure.sections():
        run_text, run_first = "", None

        def flush():
            nonlocal run_text, run_first
            if run_first is not None and run_text.strip():
                out.append(_chunk(structure, section, run_first, run_text.strip()))
            run_text, run_first = "", None

        for node in structure.own_content(section.id):
            if node.id in caption_ids:
                continue                                        # folded into its figure/table chunk below
            if node.kind is NodeKind.PARAGRAPH:
                if len(node.text) > MAX_CHARS:
                    flush()
                    out.extend(_split_long(structure, section, node))
                    continue
                if run_first is not None and len(run_text) >= MIN_CHARS and len(run_text) + len(node.text) > MAX_CHARS:
                    flush()
                if run_first is not None and len(run_text) + len(node.text) > MAX_CHARS:
                    flush()
                run_first = run_first or node
                run_text = f"{run_text}\n\n{node.text}" if run_text else node.text
            else:                                               # table, figure, equation, code: their own chunk
                flush()
                caption = node.attrs.get("caption", "")
                body = node.text if node.text != caption else ""
                text = "\n".join(x for x in (caption, body) if x)[:MAX_TABLE_CHARS]
                if text.strip():
                    out.append(_chunk(structure, section, node, text))
        flush()
    return out


def embedding_text(chunk: Chunk) -> str:
    """What is actually embedded: the text with its section path, so "Calvin cycle" matters for retrieval."""
    return f"{chunk.section_path}\n{chunk.text}" if chunk.section_path else chunk.text
