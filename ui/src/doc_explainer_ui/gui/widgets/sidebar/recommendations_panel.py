"""Suggested resources: where in THIS document to read first, and what to review.

Resources whose url is ``doc://<doc>#<section id>`` are sections of the open document: the button jumps
there. Other urls open in the browser. "To review" lists the concepts the learner is weakest on, each
with a button that asks the tutor about it.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

DOC_SCHEME = "doc://"


def parse_doc_url(url: str) -> Optional[str]:
    """``doc://<doc>#<section>`` -> the section id, else None."""
    if url.startswith(DOC_SCHEME) and "#" in url:
        return url.split("#", 1)[1] or None
    return None


def _card(*widgets) -> QFrame:
    card = QFrame()
    card.setFrameShape(QFrame.Shape.StyledPanel)
    box = QVBoxLayout(card)
    for w in widgets:
        box.addLayout(w) if hasattr(w, "addWidget") else box.addWidget(w)
    return card


class RecommendationsPanel(QWidget):
    section_requested = Signal(str)     # section id of the open document
    concept_requested = Signal(str)     # ask the tutor about this concept
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("Suggested resources")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        head.addWidget(title)
        head.addStretch()
        head.addWidget(self.refresh_button)
        root.addLayout(head)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self.body = QVBoxLayout(body)
        self.body.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(body)
        root.addWidget(scroll)
        self._resources: List[dict] = []
        self._review: List[dict] = []
        self._doc = ""
        self._explained: dict = {}            # doc id -> concept names already asked about (their card goes away)
        self._render()

    # ------------------------------------------------------------------ content
    def set_resources(self, resources: list) -> None:
        self._resources = list(resources or [])
        self._render()

    def set_review(self, beliefs: List[dict]) -> None:
        """Concepts the learner is weakest on (``name``, ``p_known``, ``evidence``)."""
        seen = [b for b in beliefs if b.get("evidence", 0) > 0 and b["p_known"] < 0.6 and b["name"] not in self._done()]
        self._review = sorted(seen, key=lambda b: b["p_known"])[:6]
        self._render()

    def set_document(self, doc_id: str) -> None:
        self._doc = doc_id

    def _done(self) -> set:
        return self._explained.setdefault(self._doc, set())

    def clear(self) -> None:
        self._resources, self._review = [], []
        self._render()

    # ------------------------------------------------------------------ drawing
    def _clear_body(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #555; margin-top: 6px;")
        return label

    def _render(self) -> None:
        self._clear_body()
        document = [r for r in self._resources if parse_doc_url(r.get("url") or "")]
        internet = [r for r in self._resources if not parse_doc_url(r.get("url") or "")]
        if not self._resources and not self._review:
            empty = QLabel("Ask for an explanation or take a quiz and suggestions appear here.")
            empty.setWordWrap(True)
            empty.setStyleSheet("color: #999; padding: 20px;")
            self.body.addWidget(empty)
            return
        if document:
            self.body.addWidget(self._section_title("Read next in this document"))
            for r in document:
                self.body.addWidget(self._resource_card(r))
        if internet:
            self.body.addWidget(self._section_title("On the internet"))
            for r in internet:
                self.body.addWidget(self._resource_card(r))
        if self._review:
            self.body.addWidget(self._section_title("To review"))
            for b in self._review:
                self.body.addWidget(self._review_card(b))

    def _resource_card(self, r: dict) -> QFrame:
        title = QLabel(r.get("title", "Resource"))
        title.setStyleSheet("font-weight: bold;")
        title.setWordWrap(True)
        desc = QLabel(r.get("description", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 10pt;")
        difficulty = getattr(r.get("difficulty"), "value", r.get("difficulty")) or "intermediate"
        source = r.get("source") or ""
        meta = QLabel(" | ".join(p for p in (source, str(r.get("type", "resource")), str(difficulty)) if p))
        meta.setStyleSheet("color: #999; font-size: 9pt;")
        widgets = [title, desc, meta]
        url = r.get("url") or ""
        section = parse_doc_url(url)
        if section:
            button = QPushButton("Go to section")
            button.clicked.connect(lambda _=False, s=section: self.section_requested.emit(s))
            widgets.append(button)
        elif url:
            button = QPushButton("Open resource")
            button.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            widgets.append(button)
        return _card(*widgets)

    def _review_card(self, b: dict) -> QFrame:
        name = QLabel(f"{b['name']}  ({int(b['p_known'] * 100)}% known)")
        name.setWordWrap(True)
        button = QPushButton("Explain this")
        button.clicked.connect(lambda _=False, n=b["name"]: self._explain(n))
        return _card(name, button)

    def _explain(self, name: str) -> None:
        """Ask about the concept, and take it off the list: it has been dealt with."""
        self._done().add(name)
        self._review = [b for b in self._review if b["name"] != name]
        self._render()
        self.concept_requested.emit(name)
