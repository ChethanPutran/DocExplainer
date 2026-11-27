from typing import Optional

import pymupdf as fitz
from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QLabel, QWidget
from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

from .document_viewer import DocumentViewer
from ...base.exceptions import ViewerError


class PDFViewer(DocumentViewer):
    """PDF document viewer"""
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.doc = None
        self.current_page_index = 0
        self.zoom = 1.0
        self.current_zoom = 1.0
        self._find_key, self._hits, self._find_index = None, [], -1
    
    def _setup_ui(self):
        """Setup PDF viewer UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for pages
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Container for pages
        self.pages_container = QWidget()
        self.pages_layout = QVBoxLayout(self.pages_container)
        self.pages_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area.setWidget(self.pages_container)
        layout.addWidget(self.scroll_area)
        
        self.setLayout(layout)
        self._attach_context_menu(self.scroll_area)
        self._attach_zoom(self.scroll_area)
    
    def load(self, path: str) -> bool:
        """Load PDF document"""
        try:
            self.doc = fitz.open(path)
            self.current_path = path
            self._render_pages()
            self.document_loaded.emit(self.doc_id)
            return True
        except Exception as e:
            raise ViewerError(f"Failed to load PDF: {e}") from e
    
    def find(self, text: str, forward: bool = True, case_sensitive: bool = False) -> bool:
        """Search every page; matches are painted on the pages and the view jumps to the current one."""
        if not self.doc:
            return False
        key = (text, case_sensitive)
        if key != getattr(self, "_find_key", None):
            self._find_key, self._find_index = key, -1
            hits = []
            for i in range(len(self.doc)):
                page = self.doc.load_page(i)
                found = page.search_for(text)
                if found and case_sensitive:
                    source = page.get_text()
                    found = found if text in source else []
                hits += [(i, r) for r in found]
            self._hits = hits
        if not self._hits:
            self.clear_find()
            return False
        step = 1 if forward else -1
        self._find_index = (self._find_index + step) % len(self._hits)
        self._paint_hits()
        page_index, rect = self._hits[self._find_index]
        container = self.pages_layout.itemAt(page_index).widget()
        self.scroll_area.verticalScrollBar().setValue(max(0, container.y() + int(rect.y0 * self.zoom) - 80))
        self.current_page = page_index + 1
        return True

    def _paint_hits(self):
        by_page = {}
        for n, (i, r) in enumerate(self._hits):
            by_page.setdefault(i, []).append((r, n == self._find_index))
        for i in range(self.pages_layout.count()):
            label = self.pages_layout.itemAt(i).widget().layout().itemAt(1).widget()
            pix = self._page_pixmap(i)
            if i in by_page:
                painter = QPainter(pix)
                for r, current in by_page[i]:
                    painter.fillRect(int(r.x0 * self.zoom), int(r.y0 * self.zoom), int((r.x1 - r.x0) * self.zoom),
                                     int((r.y1 - r.y0) * self.zoom), QColor(255, 140, 0, 120) if current else QColor(255, 235, 0, 110))
                painter.end()
            label.setPixmap(pix)

    def clear_find(self) -> None:
        self._find_key, self._hits = None, []
        if self.doc and self.pages_layout.count():
            self._paint_hits()

    def _page_pixmap(self, index: int) -> QPixmap:
        pix = self.doc.load_page(index).get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        return QPixmap.fromImage(QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy())

    def _render_pages(self):
        """Render all pages"""
        # Clear existing pages
        self._clear_pages()
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            pixmap = self._page_pixmap(page_num)
            
            # Create page label
            page_label = QLabel()
            page_label.setPixmap(pixmap)
            page_label.setAlignment(Qt.AlignCenter)
            
            # Add page number label
            page_number = QLabel(f"Page {page_num + 1}")
            page_number.setAlignment(Qt.AlignCenter)
            page_number.setStyleSheet("color: #666; font-size: 10pt;")
            
            # Container for page with number
            page_container = QWidget()
            page_container_layout = QVBoxLayout(page_container)
            page_container_layout.addWidget(page_number)
            page_container_layout.addWidget(page_label)
            
            self.pages_layout.addWidget(page_container)
    
    def _clear_pages(self):
        """Clear all rendered pages"""
        while self.pages_layout.count():
            item = self.pages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def clear(self) -> bool:
        """Clear document and free resources"""
        if self.doc:
            self.doc.close()
            self.doc = None
        
        self._clear_pages()
        self.current_path = ""
        self.current_page = 1
        self.current_position = 0
        
        return True
    
    def _page_at(self, pos) -> int:
        """1-based page under a right click (``pos`` is in the scroll area's viewport), else the current page."""
        point = self.pages_container.mapFrom(self.scroll_area.viewport(), pos)
        for i in range(self.pages_layout.count()):
            w = self.pages_layout.itemAt(i).widget()
            if w is not None and w.geometry().contains(point):
                return i + 1
        return self.current_page

    def _context_target(self, pos):
        """A rendered PDF page has no selectable text, so the menu acts on the text of the page that was clicked."""
        page = self._page_at(pos)
        text = self.doc.load_page(page - 1).get_text().strip() if self.doc and 1 <= page <= len(self.doc) else ""
        self.current_page = page
        return text, page, "page"

    @staticmethod
    def _squash(text: str) -> str:
        """Compare headings ignoring spaces and case: PDF text extraction often loses or adds spaces."""
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _heading_y(self, page: int, title: str) -> Optional[float]:
        """Vertical position (PDF points, from the top of the page) of the block that starts with the heading."""
        if not self.doc or not 1 <= page <= len(self.doc):
            return None
        wanted = self._squash(title)[:32]
        if len(wanted) < 3:
            return None
        for x0, y0, x1, y1, text, *_ in self.doc.load_page(page - 1).get_text("blocks"):
            if wanted in self._squash(text)[: len(wanted) + 40]:
                return y0
        return None

    def go_to_section(self, page: int, title: str) -> bool:
        """Scroll to the heading itself (several sections can share one page); the page top if it cannot be found."""
        if not self.go_to_page(page):
            return False
        y = self._heading_y(page, title)
        if y is not None:
            container = self.pages_layout.itemAt(page - 1).widget()
            label = container.layout().itemAt(1).widget()          # the rendered page image
            offset = container.y() + label.y() + y * self.zoom
            self.scroll_area.verticalScrollBar().setValue(max(0, int(offset) - 20))
        return True

    def go_to_page(self, page: int) -> bool:
        """Scroll to a 1-based page."""
        item = self.pages_layout.itemAt(page - 1) if 1 <= page <= self.pages_layout.count() else None
        if item is None or item.widget() is None:
            return False
        self.scroll_area.ensureWidgetVisible(item.widget(), 0, 0)
        self.scroll_area.verticalScrollBar().setValue(item.widget().y())
        self.current_page = page
        return True

    def get_selected_text(self) -> str:
        """Get selected text from current page"""
        # This would need to be implemented with text selection
        return ""
    
    def set_zoom(self, zoom: float):
        """Set zoom level"""
        self.zoom = self.current_zoom = zoom
        if self.doc:
            bar = self.scroll_area.verticalScrollBar()
            fraction = bar.value() / bar.maximum() if bar.maximum() else 0.0
            self._render_pages()
            if self._hits:
                self._paint_hits()
            self.pages_container.adjustSize()
            bar.setValue(int(fraction * bar.maximum()))
            QTimer.singleShot(0, lambda: bar.setValue(int(fraction * bar.maximum())))