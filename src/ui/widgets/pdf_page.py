from PySide6.QtWidgets import QLabel, QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QPainter, QColor, QPixmap
from .base_viewer import ActionType

class PDFPageWidget(QLabel):
    # Emit action_type and selected text
    text_action = Signal(str, str)

    def __init__(self, pixmap: QPixmap, words, scale=1.5, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)

        self.words = words  # list of [x0, y0, x1, y1, word, ...]
        self.scale = scale

        self.sel_start = None
        self.sel_end = None
        self.selection_rects = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -----------------------
    # Glyph / word under cursor
    # -----------------------
    def glyph_at_pos(self, pos):
        for i, word in enumerate(self.words):
            x0, y0, x1, y1 = [int(coord * self.scale) for coord in word[:4]]
            rect = QRect(x0, y0, x1 - x0, y1 - y0)
            if rect.contains(pos):
                return i
        return None

    # -----------------------
    # Mouse events
    # -----------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        idx = self.glyph_at_pos(event.pos())
        if idx is None:
            self.sel_start = None
            self.selection_rects.clear()
            self.update()
            return
        self.sel_start = idx
        self.sel_end = idx
        self._update_selection()

    def mouseMoveEvent(self, event):
        if self.sel_start is None or not (event.buttons() & Qt.LeftButton):
            return
        idx = self.glyph_at_pos(event.pos())
        if idx is not None:
            self.sel_end = idx
            self._update_selection()

    def mouseReleaseEvent(self, event):
        if self.sel_start is None:
            return
        self._emit_selection()

    # -----------------------
    # Update highlight rectangles
    # -----------------------
    def _update_selection(self):
        self.selection_rects.clear()
        start = min(self.sel_start, self.sel_end)
        end = max(self.sel_start, self.sel_end)
        for i in range(start, end + 1):
            x0, y0, x1, y1 = [int(coord * self.scale) for coord in self.words[i][:4]]
            self.selection_rects.append(QRect(x0, y0, x1 - x0, y1 - y0))
        self.update()

    # -----------------------
    # Emit selected text
    # -----------------------
    def _emit_selection(self):
        start = min(self.sel_start, self.sel_end)
        end = max(self.sel_start, self.sel_end)
        selected_words = [self.words[i][4] for i in range(start, end + 1)]
        text = " ".join(selected_words)
        self.text_action.emit("SELECT", text)

    # -----------------------
    # Right-click menu
    # -----------------------
    def contextMenuEvent(self, event):
        menu = QMenu(self)

        if self.sel_start is not None and self.sel_end is not None:
            # Get selected text
            start = min(self.sel_start, self.sel_end)
            end = max(self.sel_start, self.sel_end)
            selected_words = [self.words[i][4] for i in range(start, end + 1)]
            selected_text = " ".join(selected_words)

            # AI Actions
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
            # Default actions
            copy_action = QAction("Copy", self)
            select_all_action = QAction("Select All", self)

            copy_action.triggered.connect(lambda: print("Copy clicked"))
            select_all_action.triggered.connect(lambda: print("Select All clicked"))

            menu.addActions([copy_action, select_all_action])
            menu.exec(event.globalPos())

    # -----------------------
    # Paint red highlight
    # -----------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.selection_rects:
            return
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 0, 0, 100))
        for rect in self.selection_rects:
            painter.drawRect(rect)
