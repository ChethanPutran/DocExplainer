"""Job status: every document the server is processing or has processed this session, with full details.

A pure view: ``set_jobs`` takes plain dicts; it asks for fresh data with ``refresh_requested`` (every
couple of seconds while it is visible).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout)

COLUMNS = ("Document", "State", "Progress", "What it is doing")
STATE_TEXT = {"queued": "Waiting", "running": "Processing", "paused": "Paused", "succeeded": "Done", "failed": "Failed",
              "cancelled": "Cancelled"}
CAN_PAUSE, CAN_RESUME, CAN_CANCEL = ("queued", "running"), ("paused",), ("queued", "running", "paused")
KNOWLEDGE_TEXT = {"ready": "ready", "partial": "partly built", "failed": "could not be built", "none": "not built"}


def describe(job: dict) -> str:
    """Everything known about one job, as readable text."""
    lines = [f"Document:        {job.get('filename', '')}",
             f"State:           {STATE_TEXT.get(job['state'], job['state'])}"]
    if job["state"] == "queued" and job.get("queue_position"):
        lines.append(f"Queue position:  {job['queue_position']} (1 = next)")
    if job["state"] in ("queued", "running", "paused"):
        lines.append(f"Progress:        {int(job.get('progress', 0) * 100)}%")
    if job.get("stage"):
        lines.append(f"Doing now:       {job['stage']}")
    if job.get("reused"):
        lines.append("Note:            this exact file was already processed, so nothing ran again")
    lines += ["", f"Submitted:       {job.get('created_at', '')}"]
    if job.get("started_at"):
        lines.append(f"Started:         {job['started_at']}")
    if job.get("finished_at"):
        lines.append(f"Finished:        {job['finished_at']}")
    err = job.get("error")
    if err:
        lines += ["", "What went wrong", f"  {err.get('message', '')}", f"  (code: {err.get('code', '')})"]
    doc = job.get("document")
    if doc:
        lines += ["", "Document on the server",
                  f"  Title:           {doc.get('title', '')}",
                  f"  Sections:        {doc.get('sections', 0)}",
                  f"  Concepts found:  {doc.get('concepts', 0)}",
                  f"  Knowledge graph: {KNOWLEDGE_TEXT.get(doc.get('knowledge', 'none'), doc.get('knowledge'))}",
                  f"  Size:            {doc.get('size_bytes', 0):,} bytes",
                  f"  Document id:     {doc.get('doc_id', '')}"]
        if doc.get("knowledge") in ("partial", "failed"):
            lines.append("  The document can be read, but explanations are adapted to you only where the graph exists.")
    lines += ["", f"Job id:          {job.get('job_id', '')}", f"Kind:            {job.get('kind', '')}"]
    return "\n".join(lines)


class JobStatusWindow(QDialog):
    refresh_requested = Signal()
    pause_requested = Signal(str)      # job id
    resume_requested = Signal(str)
    cancel_requested = Signal(str)
    REFRESH_MS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Job status")
        self.setModal(False)
        self.resize(760, 460)
        self._jobs: List[dict] = []
        self._selected: Optional[str] = None

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("font-family: monospace;")
        self.status = QLabel()
        self.status.setStyleSheet("color: #777;")
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.details)
        refresh, close = QPushButton("Refresh now"), QPushButton("Close")
        self.pause_button, self.resume_button, self.cancel_button = QPushButton("Pause"), QPushButton("Resume"), QPushButton("Cancel job")
        self.pause_button.setToolTip("Stop at the next step and keep the work")
        self.resume_button.setToolTip("Continue where it stopped")
        self.cancel_button.setToolTip("Stop and throw the work away")
        refresh.clicked.connect(self.refresh_requested.emit)
        close.clicked.connect(self.close)
        self.pause_button.clicked.connect(lambda: self._selected and self.pause_requested.emit(self._selected))
        self.resume_button.clicked.connect(lambda: self._selected and self.resume_requested.emit(self._selected))
        self.cancel_button.clicked.connect(self._confirm_cancel)
        row = QHBoxLayout()
        row.addWidget(self.status)
        row.addStretch()
        for b in (self.pause_button, self.resume_button, self.cancel_button, refresh, close):
            row.addWidget(b)
        root = QVBoxLayout(self)
        root.addWidget(split)
        root.addLayout(row)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_requested.emit)
        self.set_jobs([])

    # ------------------------------------------------------------------ data in
    def set_jobs(self, jobs: List[dict]) -> None:
        self._jobs = sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._jobs))
        for r, job in enumerate(self._jobs):
            pct = 100 if job["state"] == "succeeded" else int(job.get("progress", 0) * 100)
            wait = f" (#{job['queue_position']})" if job["state"] == "queued" and job.get("queue_position") else ""
            cells = (job.get("filename", ""), STATE_TEXT.get(job["state"], job["state"]) + wait, f"{pct}%",
                     (job.get("error") or {}).get("message") or job.get("stage", ""))
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        keep = next((i for i, j in enumerate(self._jobs) if j["job_id"] == self._selected), 0 if self._jobs else -1)
        if keep >= 0:
            self.table.selectRow(keep)
        self.table.blockSignals(False)
        self._on_select()
        self.status.setText(f"{sum(j['state'] in ('queued', 'running') for j in self._jobs)} active, {len(self._jobs)} total"
                            if self._jobs else "No documents have been sent to the server yet.")

    def _enable_controls(self, state: Optional[str]) -> None:
        self.pause_button.setEnabled(state in CAN_PAUSE)
        self.resume_button.setEnabled(state in CAN_RESUME)
        self.cancel_button.setEnabled(state in CAN_CANCEL)

    def _confirm_cancel(self) -> None:
        if not self._selected:
            return
        job = next((j for j in self._jobs if j["job_id"] == self._selected), {})
        if QMessageBox.question(self, "Cancel job", f"Stop processing {job.get('filename', 'this document')} and discard the work done so far?"
                                ) == QMessageBox.StandardButton.Yes:
            self.cancel_requested.emit(self._selected)

    def set_error(self, message: str) -> None:
        self.status.setText(f"Could not get the job status: {message}")

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._jobs):
            self._selected = None
            self.details.setPlainText("Select a job to see its details." if self._jobs else "")
            self._enable_controls(None)
            return
        job = self._jobs[rows[0].row()]
        self._selected = job["job_id"]
        self._enable_controls(job["state"])
        bar = self.details.verticalScrollBar().value()
        self.details.setPlainText(describe(job))
        self.details.verticalScrollBar().setValue(bar)

    # ------------------------------------------------------------------ live refresh only while visible
    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start(self.REFRESH_MS)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)
