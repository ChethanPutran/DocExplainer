"""Splits a streamed reply into its text and the follow-up questions that come after a marker line.

Text is released as soon as it is certainly not part of the marker (the last ``len(marker)-1`` characters are held
back until more arrives), so the reader sees it live and never sees the marker.
"""
from __future__ import annotations

import re
from typing import List


class FollowUpSplitter:
    def __init__(self, marker: str):
        self.marker, self._buf, self._tail, self._seen_marker = marker, "", "", False
        self._released: List[str] = []

    def feed(self, piece: str) -> List[str]:
        if self._seen_marker:
            self._tail += piece
            return []
        self._buf += piece
        at = self._buf.find(self.marker)
        if at >= 0:
            self._seen_marker = True
            head, self._tail, self._buf = self._buf[:at].rstrip(), self._buf[at + len(self.marker):], ""
            return self._emit(head)
        hold = len(self.marker) - 1
        if len(self._buf) <= hold:
            return []
        out, rest = self._buf[:-hold], self._buf[-hold:]
        keep = out[len(out.rstrip()):]            # trailing whitespace may be the gap before the marker: hold it too
        self._buf = keep + rest
        return self._emit(out.rstrip())

    def finish_text(self) -> List[str]:
        """Release what was held back (the reply ended without a marker)."""
        if self._seen_marker or not self._buf:
            return []
        out, self._buf = self._buf.rstrip(), ""
        return self._emit(out)

    def _emit(self, text: str) -> List[str]:
        if not text:
            return []
        self._released.append(text)
        return [text]

    def result(self):
        """(the reply text, the follow-up questions)"""
        body = "".join(self._released).strip()
        questions = []
        for line in self._tail.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line.endswith("?") or len(line) > 3:
                questions.append(line)
        return body, questions[:3]
