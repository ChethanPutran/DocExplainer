"""The library: every document the server has processed for this user. Open, delete, refresh.

A pure view: ``set_documents`` takes plain dicts; user choices leave as signals. Deleting asks first, because it
removes the document's processed data and its conversations from the server.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

COLUMNS = ("Title", "Type", "Size", "Added", "Sections", "Concepts", "AI tutor")
KNOWLEDGE_TEXT = {"ready": "Ready", "partial": "Partly ready", "failed": "Unavailable", "none": "Not built"}


def human_size(n: int) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


class LibraryWindow(QDialog):
    refresh_requested = Signal()
    open_requested = Signal(str)       # doc id
    delete_requested = Signal(str)     # doc id
    build_graph_requested = Signal(str)  # doc id
    remove_graph_requested = Signal(str)  # doc id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Library")
        self.setModal(False)
        self.resize(860, 480)
        self._docs: List[dict] = []
        self._shown: List[dict] = []

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the library...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._fill)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda _: self._open())
        self.status = QLabel("")
        self.status.setStyleSheet("color: #777;")

        self.open_button, self.delete_button = QPushButton("Open"), QPushButton("Delete")
        self.drop_graph_button = QPushButton("Remove knowledge graph")
        self.drop_graph_button.clicked.connect(self._remove_graph)
        self.graph_button = QPushButton("Build knowledge graph")
        self.graph_button.clicked.connect(lambda: self._selected_id() and self.build_graph_requested.emit(self._selected_id()))
        refresh, close = QPushButton("Refresh"), QPushButton("Close")
        self.open_button.clicked.connect(self._open)
        self.delete_button.clicked.connect(self._delete)
        refresh.clicked.connect(self.refresh_requested.emit)
        close.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(self.status)
        row.addStretch()
        for b in (self.open_button, self.graph_button, self.drop_graph_button, self.delete_button, refresh, close):
            row.addWidget(b)
        root = QVBoxLayout(self)
        root.addWidget(self.search)
        root.addWidget(self.table)
        root.addLayout(row)
        self._update_buttons()

    # ------------------------------------------------------------------ data in
    def set_documents(self, docs: List[dict]) -> None:
        self._docs = sorted(docs, key=lambda d: d.get("uploaded_at", ""), reverse=True)
        self._fill()

    def set_error(self, message: str) -> None:
        self.status.setText(f"Could not load the library: {message}")

    def _selected_id(self) -> Optional[str]:
        row = self.table.currentRow()
        return self._shown[row]["doc_id"] if 0 <= row < len(self._shown) else None

    def _fill(self, *_):
        keep = self._selected_id()
        needle = self.search.text().strip().lower()
        self._shown = [d for d in self._docs if needle in f"{d.get('title', '')} {d.get('filename', '')}".lower()]
        self.table.setRowCount(len(self._shown))
        for r, d in enumerate(self._shown):
            cells = (d.get("title") or d.get("filename", ""), os.path.splitext(d.get("filename", ""))[1].lstrip(".").upper(),
                     human_size(d.get("size_bytes", 0)), (d.get("uploaded_at", "") or "")[:16].replace("T", " "),
                     str(d.get("sections", 0)), str(d.get("concepts", 0)),
                     KNOWLEDGE_TEXT.get(d.get("knowledge", "none"), d.get("knowledge", "")))
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        row = next((i for i, d in enumerate(self._shown) if d["doc_id"] == keep), 0 if self._shown else -1)
        if row >= 0:
            self.table.selectRow(row)
        total = sum(d.get("size_bytes", 0) for d in self._docs)
        self.status.setText((f"{len(self._shown)} of {len(self._docs)} documents" if needle else f"{len(self._docs)} documents")
                            + (f", {human_size(total)}" if self._docs else "") if self._docs
                            else "No documents yet. Open a file and it will appear here.")
        self._update_buttons()

    def _update_buttons(self) -> None:
        doc_id = self._selected_id()
        has = doc_id is not None
        self.open_button.setEnabled(has)
        self.delete_button.setEnabled(has)
        status = next((d.get("knowledge", "none") for d in self._shown if d["doc_id"] == doc_id), None)
        self.graph_button.setEnabled(status in ("none", "failed", "partial"))
        self.drop_graph_button.setEnabled(status in ("ready", "partial", "failed"))
        self.drop_graph_button.setToolTip("Stops adaptive explanations and knowledge tracking for this document. The document stays.")
        self.graph_button.setText("Finish knowledge graph" if status in ("failed", "partial") else "Build knowledge graph")
        self.graph_button.setToolTip("Already built." if status == "ready" else
                                     "Adds adaptive explanations, quizzes and knowledge tracking to this document.")

    # ------------------------------------------------------------------ user choices
    def _open(self) -> None:
        doc_id = self._selected_id()
        if doc_id:
            self.open_requested.emit(doc_id)

    def _remove_graph(self) -> None:
        doc_id = self._selected_id()
        if not doc_id:
            return
        answer = QMessageBox.question(
            self, "Remove knowledge graph",
            "Remove the knowledge graph of this document?\n\nExplanations stop adapting to what you know, and quizzes and your "
            "knowledge tracking for it are removed. You can still read it and ask questions, and rebuild the graph later.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.remove_graph_requested.emit(doc_id)

    def _delete(self) -> None:
        doc_id = self._selected_id()
        if not doc_id:
            return
        doc = next(d for d in self._shown if d["doc_id"] == doc_id)
        answer = QMessageBox.question(
            self, "Delete from library",
            f"Delete “{doc.get('title') or doc.get('filename', '')}” from the library?\n\n"
            "Its processed data and its conversations and notes are removed. The original file on your computer is not touched.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(doc_id)
