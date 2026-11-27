from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QToolButton, QWidget


class FindBar(QWidget):
    """Search box under the document: Enter = next, Shift+Enter = previous, Esc = close."""
    find_requested = Signal(str, bool, bool)      # text, forward, case sensitive
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("findBar")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Find in document")
        self.edit.setClearButtonEnabled(True)
        self.case = QCheckBox("Match case")
        self.result = QLabel("")
        self.prev_button, self.next_button, self.close_button = QToolButton(), QToolButton(), QToolButton()
        for b, text, tip in ((self.prev_button, "▲", "Previous (Shift+Enter)"), (self.next_button, "▼", "Next (Enter)"),
                             (self.close_button, "✕", "Close (Esc)")):
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        for w in (self.edit, self.prev_button, self.next_button, self.case, self.result, self.close_button):
            row.addWidget(w)
        row.setStretch(0, 1)
        self.edit.returnPressed.connect(self._enter)
        self.edit.textChanged.connect(lambda: self._go(True))
        self.next_button.clicked.connect(lambda: self._go(True))
        self.prev_button.clicked.connect(lambda: self._go(False))
        self.case.toggled.connect(lambda: self._go(True))
        self.close_button.clicked.connect(self.close_bar)
        self.hide()

    def _enter(self):
        from PySide6.QtWidgets import QApplication
        self._go(not (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier))

    def _go(self, forward: bool):
        if self.edit.text():
            self.find_requested.emit(self.edit.text(), forward, self.case.isChecked())
        else:
            self.set_result(None)

    def open_bar(self):
        self.show()
        self.edit.setFocus()
        self.edit.selectAll()

    def close_bar(self):
        self.hide()
        self.closed.emit()

    def set_result(self, found):
        self.result.setText("" if found is None else ("" if found else "No matches"))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close_bar()
        else:
            super().keyPressEvent(event)
