"""Sentence segmentation: pure Python, deterministic, no model to download or load."""
from __future__ import annotations

import re
from typing import List

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig", "figs", "eq", "eqs",
    "no", "vol", "pp", "al", "cf", "approx", "inc", "ltd", "co", "dept", "est", "ca", "sec", "ch", "ed", "eds",
}
# a place a sentence could end: terminal punctuation (+ closing quotes/brackets), whitespace, then a capital/digit/opener
_BOUNDARY = re.compile(r"""([.!?]+["')\]]*)\s+(?=[A-Z0-9"'(\[])""")
_SPACE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences. Keeps abbreviations ("Dr. Smith", "e.g. this"), initials ("A. Turing")
    and decimals ("3.14") together."""
    text = normalize_space(text)
    if not text:
        return []
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        before = text[start:m.start(1) + 1]
        words = before.split()
        last = words[-1].rstrip(".!?\"')]").lower() if words else ""
        punct = m.group(1)
        if punct.startswith(".") and (last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())
                                      or re.fullmatch(r"(?:[a-z]\.){1,}[a-z]?", last + ".")):
            continue
        out.append(text[start:m.end(1)].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
