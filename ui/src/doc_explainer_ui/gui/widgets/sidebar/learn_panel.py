"""The Learn tab: knowledge-build status, where the reader stands, the diagnostic and practice quizzes.

Pure view: it shows plain dicts and emits signals; the window manager talks to the service.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QRadioButton, QStackedWidget, QVBoxLayout, QWidget)

OVERVIEW, SETUP, QUIZ = 0, 1, 2
_NEEDS_GRAPH = "Quizzes need the knowledge graph."
_STATUS_TEXT = {
    "ready": "Knowledge graph ready",
    "partial": "Knowledge graph partly built: explanations are adapted only where it exists",
    "failed": "Knowledge graph could not be built: explanations are not adapted to you",
    "none": "No knowledge graph for this document",
}


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold; font-size: 12pt;")
    return label


def _rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _wrapped(text: str = "", style: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    if style:
        label.setStyleSheet(style)
    return label


class LearnPanel(QWidget):
    refresh_requested = Signal()
    diagnostic_requested = Signal(str, list)          # level, known topics
    practice_requested = Signal()
    answer_submitted = Signal(str, int)               # question id, chosen option
    finished_section = Signal()                       # "I have read the section I was last asking about"
    quiz_finished = Signal(str)                       # "diagnostic" | "practice"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc_id = ""
        self._questions: List[dict] = []
        self._index = 0
        self._kind = "practice"
        self._answered = False
        self._build()
        self.clear()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(_heading("Learn"))
        root.addWidget(_rule())
        self.stack = QStackedWidget()
        root.addWidget(self.stack)
        self.stack.addWidget(self._overview_page())
        self.stack.addWidget(self._setup_page())
        self.stack.addWidget(self._quiz_page())

    def _overview_page(self) -> QWidget:
        page, box = QWidget(), QVBoxLayout()
        page.setLayout(box)
        self.status_label = _wrapped()
        self.build_bar = QProgressBar()
        self.build_bar.setRange(0, 100)
        self.stage_label = _wrapped(style="color: #666; font-size: 9pt;")
        self.level_label = _wrapped(style="font-weight: bold;")
        self.strong_label = _wrapped(style="color: #28a745;")
        self.weak_label = _wrapped(style="color: #dc3545;")
        self.message = _wrapped(style="color: #b35c00;")
        for w in (self.status_label, self.build_bar, self.stage_label, _rule(), self.level_label,
                  self.strong_label, self.weak_label):
            box.addWidget(w)
        row = QHBoxLayout()
        self.diagnostic_button = QPushButton("Check my level")
        self.diagnostic_button.setToolTip("A few questions so explanations start at the right level")
        self.practice_button = QPushButton("Practice quiz")
        self.refresh_button = QPushButton("Refresh")
        for b in (self.diagnostic_button, self.practice_button, self.refresh_button):
            row.addWidget(b)
        box.addLayout(row)
        self.finished_button = QPushButton("I finished this section")
        self.finished_button.setToolTip("Tells the tutor you have read the section you were last asking about")
        box.addWidget(self.finished_button)
        self.finished_button.clicked.connect(self.finished_section.emit)
        box.addWidget(self.message)
        box.addStretch()
        self.diagnostic_button.clicked.connect(lambda: self.stack.setCurrentIndex(SETUP))
        self.practice_button.clicked.connect(self.practice_requested.emit)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        return page

    def _setup_page(self) -> QWidget:
        page, box = QWidget(), QVBoxLayout()
        page.setLayout(box)
        box.addWidget(_wrapped("Tell us where you stand. A few questions will then fine-tune it."))
        self.level_combo = QComboBox()
        self.level_combo.addItems(["beginner", "intermediate", "advanced"])
        self.level_combo.setCurrentText("intermediate")
        self.topics_edit = QLineEdit()
        self.topics_edit.setPlaceholderText("Topics you already know, separated by commas (optional)")
        row = QHBoxLayout()
        start, back = QPushButton("Start"), QPushButton("Back")
        row.addWidget(start)
        row.addWidget(back)
        for w in (QLabel("Your level"), self.level_combo, QLabel("Already know"), self.topics_edit):
            box.addWidget(w)
        box.addLayout(row)
        box.addStretch()
        start.clicked.connect(self._start_diagnostic)
        back.clicked.connect(lambda: self.stack.setCurrentIndex(OVERVIEW))
        return page

    def _quiz_page(self) -> QWidget:
        page, box = QWidget(), QVBoxLayout()
        page.setLayout(box)
        self.quiz_title = _wrapped(style="color: #666; font-size: 9pt;")
        self.concept_label = _wrapped(style="font-weight: bold;")
        self.question_label = _wrapped()
        self.options_box = QVBoxLayout()
        self.group = QButtonGroup(self)
        self.result_label = _wrapped()
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        row = QHBoxLayout()
        self.submit_button, self.next_button, self.leave_button = QPushButton("Submit"), QPushButton("Next"), QPushButton("Leave quiz")
        for b in (self.submit_button, self.next_button, self.leave_button):
            row.addWidget(b)
        for w in (self.quiz_title, self.concept_label, self.question_label):
            box.addWidget(w)
        box.addLayout(self.options_box)
        box.addWidget(self.result_label)
        box.addLayout(row)
        box.addStretch()
        self.submit_button.clicked.connect(self._submit)
        self.next_button.clicked.connect(self._next)
        self.leave_button.clicked.connect(self._leave)
        return page

    # ------------------------------------------------------------------ state from the service
    def clear(self) -> None:
        self.doc_id = ""
        self._questions, self._index = [], 0
        self.status_label.setText("Open a document to see what the tutor knows about it.")
        self.build_bar.hide()
        for label in (self.stage_label, self.level_label, self.strong_label, self.weak_label, self.message):
            label.setText("")
        self._enable_actions(False)
        self.stack.setCurrentIndex(OVERVIEW)

    def set_document(self, doc_id: str) -> None:
        if doc_id != self.doc_id:
            self.clear()
            self.doc_id = doc_id

    def _enable_actions(self, on: bool) -> None:
        for b in (self.diagnostic_button, self.practice_button, self.refresh_button, self.finished_button):
            b.setEnabled(on)

    def set_building(self, progress: float, stage: str) -> None:
        """Registration progress (0..1) while the document is processed."""
        self.build_bar.show()
        self.build_bar.setValue(int(max(0.0, min(1.0, progress)) * 100))
        self.status_label.setText("Preparing the document...")
        self.stage_label.setText(stage)
        self._enable_actions(False)

    def set_knowledge(self, status: str, sections: int, concepts: int) -> None:
        self.build_bar.hide()
        self.stage_label.setText("")
        text = _STATUS_TEXT.get(status, status)
        self.status_label.setText(f"{text}\n{concepts} concepts in {sections} sections" if concepts else text)
        usable = status in ("ready", "partial")
        self._enable_actions(usable)
        # Only manage the message this method owns: a quiz error that arrived meanwhile must not be wiped.
        if not usable:
            self.message.setText(_NEEDS_GRAPH)
        elif self.message.text() == _NEEDS_GRAPH:
            self.message.setText("")

    def set_learner(self, level: str, beliefs: List[dict]) -> None:
        self.level_label.setText(f"Your level: {level}")
        seen = sorted((b for b in beliefs if b.get("evidence", 0) > 0), key=lambda b: b["p_known"])
        weak = [b["name"] for b in seen if b["p_known"] < 0.4][:5]
        strong = [b["name"] for b in reversed(seen) if b["p_known"] >= 0.7][:5]
        self.strong_label.setText("Comfortable with: " + ", ".join(strong) if strong else "")
        self.weak_label.setText("Needs work: " + ", ".join(weak) if weak else "")

    def set_message(self, text: str) -> None:
        self.message.setText(text)

    # ------------------------------------------------------------------ quiz flow
    def _start_diagnostic(self) -> None:
        topics = [t.strip() for t in self.topics_edit.text().split(",") if t.strip()]
        self.set_message("Preparing questions...")
        self.stack.setCurrentIndex(OVERVIEW)
        self._enable_actions(False)
        self.diagnostic_requested.emit(self.level_combo.currentText(), topics)

    def show_questions(self, questions: List[dict], kind: str) -> None:
        self._enable_actions(True)
        if not questions:
            self.set_message("No questions could be made for this document yet.")
            self.stack.setCurrentIndex(OVERVIEW)
            return
        self.message.setText("")
        self._questions, self._index, self._kind = list(questions), 0, kind
        self._show_current()
        self.stack.setCurrentIndex(QUIZ)

    def _show_current(self) -> None:
        q = self._questions[self._index]
        self._answered = False
        what = "Level check" if self._kind == "diagnostic" else "Practice"
        self.quiz_title.setText(f"{what}: question {self._index + 1} of {len(self._questions)}")
        self.concept_label.setText(q.get("concept", ""))
        self.question_label.setText(q["question"])
        for b in list(self.group.buttons()):
            self.group.removeButton(b)
            b.deleteLater()
        for i, option in enumerate(q["options"]):
            button = QRadioButton(option)
            button.setStyleSheet("QRadioButton { padding: 2px; }")
            self.group.addButton(button, i)
            self.options_box.addWidget(button)
        self.result_label.setText("")
        self.submit_button.setEnabled(True)
        self.next_button.hide()
        self.submit_button.show()

    def _submit(self) -> None:
        choice = self.group.checkedId()
        if choice < 0:
            self.result_label.setText("Pick an answer first.")
            return
        self.submit_button.setEnabled(False)
        self.answer_submitted.emit(self._questions[self._index]["id"], choice)

    def show_answer(self, result: dict, chosen: int | None = None) -> None:
        """Mark the answer. Called once the service has graded it."""
        self._answered = True
        right = result.get("correct_index", -1)
        for b in self.group.buttons():
            b.setEnabled(False)
            if self.group.id(b) == right:
                b.setStyleSheet("QRadioButton { color: #28a745; font-weight: bold; }")
        verdict = ("<b style='color:#28a745'>Correct.</b>" if result.get("correct")
                   else "<b style='color:#dc3545'>Not quite.</b>")
        self.result_label.setText(f"{verdict} {result.get('explanation', '')}")
        last = self._index + 1 >= len(self._questions)
        self.next_button.setText("Finish" if last else "Next")
        self.submit_button.hide()
        self.next_button.show()

    def show_answer_error(self, message: str) -> None:
        self.result_label.setText(message)
        self.submit_button.setEnabled(True)

    def _next(self) -> None:
        if self._index + 1 < len(self._questions):
            self._index += 1
            self._show_current()
        else:
            self._leave()

    def _leave(self) -> None:
        finished = self._answered or self._index > 0
        kind = self._kind
        self._questions, self._index = [], 0
        self.stack.setCurrentIndex(OVERVIEW)
        if finished:
            self.quiz_finished.emit(kind)
