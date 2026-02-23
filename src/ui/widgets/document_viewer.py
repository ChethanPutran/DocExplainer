from PySide6.QtWidgets import QTextBrowser, QMenu
from PySide6.QtGui import QAction
from .base_viewer import BaseViewer, ActionType

class DocumentViewer(QTextBrowser, BaseViewer):
    def __init__(self, parent=None):
        QTextBrowser.__init__(self, parent)
        BaseViewer.__init__(self)
        self.setReadOnly(True)

    def load(self, path):
        # Assuming load_document returns HTML or plain text
        from ..utils.utils import load_document
        self.setHtml(load_document(path))

    def _get_metadata(self):
        cursor = self.textCursor()
        pos = cursor.selectionStart()
        if self.doc_model:
            for section in self.doc_model.sections:
                for para in section.paragraphs:
                    if para.start <= pos <= para.end:
                        return para.page, pos
        return 1, pos

    def contextMenuEvent(self, event):
        text = self.textCursor().selectedText().strip()
        if not text:
            return super().contextMenuEvent(event)

        menu = QMenu(self)
        explain = menu.addAction("Explain Text")
        summarize = menu.addAction("Summarize")
        
        page, pos = self._get_metadata()
        action = menu.exec(event.globalPos())
        
        if action == explain:
            self.text_action.emit(ActionType.EXPLAIN.value, self.doc_id, text, page, pos)
        elif action == summarize:
            self.text_action.emit(ActionType.SUMMARIZE.value, self.doc_id, text, page, pos)

    def clear(self):
        super().clear()
        self.doc_id = None