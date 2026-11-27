"""Cutting the document into the steps along which knowledge is built: one per section, in reading order.

A very long section is split into windows so no single request is huge. Only a section's OWN content
counts for its step (its subsections are steps of their own, later), so a step never sees text the
reader would not have reached yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..common.structure import DocumentStructure, NodeKind

MAX_STEP_WORDS = 1500
MAX_TABLE_CHARS = 1500


@dataclass
class Step:
    index: int
    section_id: str
    title: str
    path: str
    ordinal_start: int
    ordinal_end: int
    paragraphs: List[Tuple[str, int, str]]       # (node id, ordinal, text), numbered 1..n in prompts

    @property
    def words(self) -> int:
        return sum(len(t.split()) for _, _, t in self.paragraphs)


def plan_steps(structure: DocumentStructure, max_words: int = MAX_STEP_WORDS) -> List[Step]:
    """The deterministic list of steps for this document (same structure -> same steps)."""
    caption_ids = {e.src for e in structure.edges if e.kind.value == "captions"}
    steps: List[Step] = []
    for section in structure.sections():
        items: List[Tuple[str, int, str]] = []
        for node in structure.own_content(section.id):
            if node.kind is NodeKind.PARAGRAPH:
                text = node.text
            elif node.kind in (NodeKind.TABLE, NodeKind.FIGURE):
                text = "\n".join(x for x in (node.attrs.get("caption", ""), node.text if node.text != node.attrs.get("caption") else "") if x)[:MAX_TABLE_CHARS]
            else:
                continue                                   # equations and code add nothing to concepts here
            if node.id in caption_ids or not text.strip():
                continue
            items.append((node.id, node.ordinal, text))
        if not items:
            continue
        window: List[Tuple[str, int, str]] = []
        words = 0
        path = " > ".join(structure.section_path(section.id))
        for item in items:
            n = len(item[2].split())
            if window and words + n > max_words:
                steps.append(_make(len(steps), section, path, window))
                window, words = [], 0
            window.append(item)
            words += n
        if window:
            steps.append(_make(len(steps), section, path, window))
    return steps


def _make(index, section, path, window) -> Step:
    return Step(index=index, section_id=section.id, title=section.title, path=path,
                ordinal_start=window[0][1], ordinal_end=window[-1][1], paragraphs=list(window))
