"""Turns a reply written with Markdown into what a person would actually say aloud."""
from __future__ import annotations

import re

_RULES = [
    (re.compile(r"```.*?```", re.S), " code omitted. "),
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),                # image -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),                 # link -> its words
    (re.compile(r"https?://\S+"), " link "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),                  # headings
    (re.compile(r"^\s*>\s?", re.M), ""),                           # quotes
    (re.compile(r"^\s*([-*+•]|\d+[.)])\s+", re.M), ""),            # list markers
    (re.compile(r"^\s*([-*_=]\s*){3,}$", re.M), " "),              # rules
    (re.compile(r"(\*\*|__|\*|~~)"), ""),                          # emphasis marks
    (re.compile(r"(?<=\w)_(?=\w)"), " "),                          # snake_case
    (re.compile(r"\|"), ", "),                                     # table cells
    (re.compile(r"[#*_`~^<>\[\]{}\\]"), " "),                      # anything symbolic left over
    (re.compile(r"\s*\n\s*"), ". "),
    (re.compile(r"([.!?:;,])(\s*\1)+"), r"\1"),
    (re.compile(r"\.\s*([,;:])"), r"\1"),
    (re.compile(r"[ \t]{2,}"), " "),
]
_SPOKEN = {"&": " and ", "%": " percent", "=": " equals ", "+": " plus ", "->": " to ", "→": " to ", "≈": " approximately ",
           "≤": " at most ", "≥": " at least ", "×": " times ", "e.g.": "for example", "i.e.": "that is", "etc.": "and so on"}


def speakable(text: str) -> str:
    for pattern, repl in _RULES[:9]:
        text = pattern.sub(repl, text)
    for symbol, words in _SPOKEN.items():
        text = text.replace(symbol, words)
    for pattern, repl in _RULES[9:]:
        text = pattern.sub(repl, text)
    return text.strip(" \t,;")
