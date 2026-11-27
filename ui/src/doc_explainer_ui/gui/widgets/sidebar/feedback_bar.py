from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

VERDICTS = (("Too hard", "too_hard"), ("Just right", "just_right"), ("Too easy", "too_easy"))


class FeedbackBar(QWidget):
    """"Was that explanation pitched right?" One click tunes the next ones."""
    verdict_given = Signal(str)   # too_hard | just_right | too_easy

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 2, 6, 2)
        self.label = QLabel("This explanation was:")
        row.addWidget(self.label)
        self.buttons = {}
        for text, verdict in VERDICTS:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, v=verdict: self._give(v))
            row.addWidget(b)
            self.buttons[verdict] = b
        row.addStretch()
        self.reset(False)

    def _give(self, verdict: str) -> None:
        self.set_enabled(False)
        self.label.setText("Thanks, noted.")
        self.verdict_given.emit(verdict)

    def set_enabled(self, on: bool) -> None:
        for b in self.buttons.values():
            b.setEnabled(on)

    def reset(self, on: bool) -> None:
        self.label.setText("This explanation was:")
        self.set_enabled(on)
