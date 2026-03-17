import fitz  # PyMuPDF
from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QLabel, QWidget
from PySide6.QtCore import Qt, QRect
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
        
        self._setup_ui()
    
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
    
    def _render_pages(self):
        """Render all pages"""
        # Clear existing pages
        self._clear_pages()
        
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
            
            # Convert to QImage
            img = QImage(pix.samples, pix.width, pix.height,
                        pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            
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
    
    def get_selected_text(self) -> str:
        """Get selected text from current page"""
        # This would need to be implemented with text selection
        return ""
    
    def set_zoom(self, zoom: float):
        """Set zoom level"""
        self.zoom = zoom
        if self.doc:
            self._render_pages()