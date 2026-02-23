import fitz
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QMenu
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QAction
from PySide6.QtCore import Qt, Signal, QRect
from .base_viewer import BaseViewer, ActionType

class PDFPageWidget(QLabel):
    # Signals to parent: action, text, page_num
    page_text_action = Signal(str, str, int)

    def __init__(self, pixmap: QPixmap, words, page_num: int, scale=1.5, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.words = words
        self.page_num = page_num
        self.scale = scale
        self.sel_start = None
        self.sel_end = None
        self.selection_rects = []
        self.setMouseTracking(True)

    def glyph_at_pos(self, pos):
        for i, word in enumerate(self.words):
            x0, y0, x1, y1 = [int(coord * self.scale) for coord in word[:4]]
            if QRect(x0, y0, x1 - x0, y1 - y0).contains(pos):
                return i
        return None

    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            idx = self.glyph_at_pos(event.pos())
            self.sel_start = idx
            self.sel_end = idx
            self._update_selection()

    def mouseMoveEvent(self, event):
        if self.sel_start is not None and (event.buttons() & Qt.LeftButton):
            idx = self.glyph_at_pos(event.pos())
            if idx is not None:
                self.sel_end = idx
                self._update_selection()

    def _update_selection(self):
        self.selection_rects.clear()
        if self.sel_start is None: return
        start, end = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
        for i in range(start, end + 1):
            x0, y0, x1, y1 = [int(coord * self.scale) for coord in self.words[i][:4]]
            self.selection_rects.append(QRect(x0, y0, x1 - x0, y1 - y0))
        self.update()

    def get_selected_text(self):
        if self.sel_start is None: return ""
        start, end = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
        return " ".join([self.words[i][4] for i in range(start, end + 1)])

    def contextMenuEvent(self, event):
        text = self.get_selected_text()
        if not text: return
        
        menu = QMenu(self)
        explain = menu.addAction("Explain Text")
        summarize = menu.addAction("Summarize")
        
        action = menu.exec(event.globalPos())
        if action == explain:
            self.page_text_action.emit(ActionType.EXPLAIN.value, text, self.page_num)
        elif action == summarize:
            self.page_text_action.emit(ActionType.SUMMARIZE.value, text, self.page_num)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rects:
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0, 80))
            painter.setPen(Qt.NoPen)
            for rect in self.selection_rects:
                painter.drawRect(rect)

class PDFViewer(QWidget, BaseViewer):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        BaseViewer.__init__(self)
        
        self.layout_ = QVBoxLayout(self)
        self.scroll_ = QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.scroll_.setWidget(self.container)
        self.layout_.addWidget(self.scroll_)

    def load(self, path):
        self.clear()
        self.doc = fitz.open(path)
        for i, page in enumerate(self.doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            
            page_widget = PDFPageWidget(QPixmap.fromImage(img), page.get_text("words"), page_num=i+1)
            page_widget.page_text_action.connect(self._handle_page_signal)
            self.container_layout.addWidget(page_widget)

    def _handle_page_signal(self, action, text, page_num):
        self.last_selection = text
        self.text_action.emit(action, self.doc_id, text, page_num, 0)

    def clear(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if self.doc:
            self.doc.close()
            self.doc = None