from PySide6.QtWidgets import QTextBrowser, QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal

from ..utils.utils import load_document
from .base_viewer import BaseViewer, ActionType


class DocumentViewer(QTextBrowser, BaseViewer):

    text_action = Signal(str, str)  # action, selected_text

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def load(self, path):
        content = load_document(path)
        self.setHtml(content)

    def contextMenuEvent(self, event):
        cursor = self.textCursor()
        selected_text = cursor.selectedText().strip()

        menu = QMenu(self)

        if selected_text:
            explain = QAction("Explain Text", self)
            summarize = QAction("Summarize", self)
            ask = QAction("Ask Question", self)

            menu.addActions([explain, summarize])
            menu.addSeparator()
            menu.addAction(ask)

            action = menu.exec(event.globalPos())

            if action == explain:
                self.text_action.emit(ActionType.EXPLAIN.value, selected_text)
            elif action == summarize:
                self.text_action.emit(ActionType.SUMMARIZE.value, selected_text)
            elif action == ask:
                self.text_action.emit(ActionType.ASK.value, selected_text)
        else:
            copy_action = QAction("Copy", self)
            select_all_action = QAction("Select All", self)

            copy_action.triggered.connect(self.copy)
            select_all_action.triggered.connect(self.selectAll)

            menu.addActions([copy_action, select_all_action])
            menu.exec(event.globalPos())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        selected_text = self.textCursor().selectedText().strip()
        if selected_text:
            self.text_action.emit(ActionType.RELEASE.value, selected_text)

    def clear(self):
        super().clear()
        self.setHtml("<h2>Load a document here</h2>")
