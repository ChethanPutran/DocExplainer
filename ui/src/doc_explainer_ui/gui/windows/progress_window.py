"""Progress dashboard: where the reader stands, how it changed over time, and what to review next.

A pure view: ``set_data`` takes plain dicts (learner, history points, review items).
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


def mastery_series(points: List[dict]) -> List[tuple]:
    """(time, average belief over every concept practised so far) after each event, oldest first."""
    latest: Dict[str, float] = {}
    series = []
    for p in sorted(points, key=lambda x: x["at"]):
        latest[p["concept_id"]] = p["after"]
        series.append((p["at"], sum(latest.values()) / len(latest)))
    return series


class MasteryChart(QWidget):
    """Average mastery over time, drawn directly (no chart library)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series: List[tuple] = []
        self.setMinimumHeight(170)

    def set_series(self, series: List[tuple]) -> None:
        self.series = series
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text, grid, accent = pal.windowText().color(), pal.mid().color(), pal.highlight().color()
        left, top, right, bottom = 38, 10, 12, 24
        area = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        p.setPen(QPen(grid, 1))
        for frac in (0, 0.5, 1):
            y = area.bottom() - frac * area.height()
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(text)
            p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{int(frac * 100)}%")
            p.setPen(QPen(grid, 1))
        if len(self.series) < 2:
            p.setPen(text)
            p.drawText(area, Qt.AlignmentFlag.AlignCenter,
                       "Answer a few quiz questions to see your progress here." if not self.series else "One more answer and the line appears.")
            return
        t0, t1 = self.series[0][0], self.series[-1][0]
        span = max(t1 - t0, 1e-6)
        pts = [QPointF(area.left() + (t - t0) / span * area.width(), area.bottom() - v * area.height()) for t, v in self.series]
        fill = QPolygonF([QPointF(pts[0].x(), area.bottom()), *pts, QPointF(pts[-1].x(), area.bottom())])
        soft = QColor(accent)
        soft.setAlpha(50)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(soft)
        p.drawPolygon(fill)
        p.setPen(QPen(accent, 2.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(QPolygonF(pts))
        p.setPen(text)
        fmt = "%d %b" if t1 - t0 > 86400 else "%H:%M"
        p.drawText(QRectF(area.left(), area.bottom() + 4, 80, 16), Qt.AlignmentFlag.AlignLeft, dt.datetime.fromtimestamp(t0).strftime(fmt))
        p.drawText(QRectF(area.right() - 80, area.bottom() + 4, 80, 16), Qt.AlignmentFlag.AlignRight, dt.datetime.fromtimestamp(t1).strftime(fmt))


class ProgressWindow(QDialog):
    refresh_requested = Signal()
    practise_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Your progress")
        self.setModal(False)
        self.resize(720, 640)
        self.level = QLabel("")
        self.level.setStyleSheet("font-size: 15pt; font-weight: 600;")
        self.summary = QLabel("")
        self.chart = MasteryChart()
        self.concepts = QTableWidget(0, 3)
        self.concepts.setHorizontalHeaderLabels(("Concept", "How well you know it", "Practised"))
        self.review = QTableWidget(0, 3)
        self.review.setHorizontalHeaderLabels(("Review soon", "Remembered", "Last practised"))
        for t in (self.concepts, self.review):
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            t.verticalHeader().hide()
            t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status = QLabel("")
        refresh, practise, close = QPushButton("Refresh"), QPushButton("Practise what is due"), QPushButton("Close")
        self.practise_button = practise
        refresh.clicked.connect(self.refresh_requested.emit)
        practise.clicked.connect(self.practise_requested.emit)
        close.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(self.status)
        row.addStretch()
        for b in (practise, refresh, close):
            row.addWidget(b)
        root = QVBoxLayout(self)
        root.addWidget(self.level)
        root.addWidget(self.summary)
        root.addWidget(QLabel("Average mastery over time"))
        root.addWidget(self.chart)
        root.addWidget(QLabel("Concepts"))
        root.addWidget(self.concepts, 3)
        root.addWidget(self.review, 2)
        root.addLayout(row)

    def set_error(self, message: str) -> None:
        self.status.setText(f"Could not load your progress: {message}")

    def set_data(self, learner: Optional[dict], history: List[dict], review: List[dict]) -> None:
        self.status.setText("")
        if not learner:
            self.level.setText("No knowledge tracking for this document")
            self.summary.setText("Build the knowledge graph from the Library to track what you know.")
            self.chart.set_series([])
            self.concepts.setRowCount(0)
            self.review.setRowCount(0)
            self.practise_button.setEnabled(False)
            return
        beliefs = sorted(learner.get("concepts", []), key=lambda b: (-(b.get("evidence", 0) > 0), b["p_known"]))
        seen = [b for b in beliefs if b.get("evidence", 0) > 0]
        self.level.setText(f"Your level: {str(learner.get('level', '')).capitalize()}")
        known = sum(b["p_known"] >= 0.7 for b in seen)
        self.summary.setText(f"{len(seen)} concepts practised, {known} you know well, "
                             f"{sum(r['due'] for r in review)} due for review.")
        self.chart.set_series(mastery_series(history))
        self.concepts.setRowCount(len(beliefs))
        for r, b in enumerate(beliefs):
            self.concepts.setItem(r, 0, QTableWidgetItem(b["name"]))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(b["p_known"] * 100)))
            bar.setTextVisible(True)
            self.concepts.setCellWidget(r, 1, bar)
            self.concepts.setItem(r, 2, QTableWidgetItem(str(b.get("evidence", 0))))
        self.concepts.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        due = [i for i in review if i["due"]] or review[:5]
        self.review.setRowCount(len(due))
        for r, i in enumerate(due):
            self.review.setItem(r, 0, QTableWidgetItem(i["name"]))
            self.review.setItem(r, 1, QTableWidgetItem(f"{int(round(i['retention'] * 100))}%"))
            self.review.setItem(r, 2, QTableWidgetItem(dt.datetime.fromtimestamp(i["last_seen"]).strftime("%d %b %H:%M")))
        self.practise_button.setEnabled(any(i["due"] for i in review))
