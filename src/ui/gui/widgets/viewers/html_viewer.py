from PySide6.QtWidgets import QTextBrowser, QVBoxLayout
from PySide6.QtCore import QUrl, QTimer, Signal
from PySide6.QtGui import QTextCursor

from .document_viewer import DocumentViewer
from ...utils.file_utils import FileUtils


class HTMLViewer(DocumentViewer):
    """HTML document viewer"""
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.content = ""
        self._setup_ui()
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._emit_selection)
    
    def _setup_ui(self):
        """Setup HTML viewer UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setOpenLinks(True)
        
        # Connect signals
        self.browser.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.browser.selectionChanged.connect(self._on_selection_changed)
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        
        layout.addWidget(self.browser)
        self.setLayout(layout)
    
    def load(self, path: str) -> bool:
        """Load HTML document"""
        try:
            # Try to load as URL first
            if path.startswith(('http://', 'https://')):
                self.browser.setSource(QUrl(path))
            else:
                # Load local file
                self.content = FileUtils.read_text_file(path)
                if self.content is None:
                    return False
                self.browser.setHtml(self.content)
            
            self.current_path = path
            self.document_loaded.emit(self.doc_id)
            return True
            
        except Exception as e:
            print(f"Error loading HTML file: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear document"""
        self.browser.clear()
        self.content = ""
        self.current_path = ""
        self.current_page = 1
        self.current_position = 0
        return True
    
    def get_selected_text(self) -> str:
        """Get selected text"""
        cursor = self.browser.textCursor()
        return cursor.selectedText()
    
    def get_current_position(self) -> int:
        """Get current text position"""
        cursor = self.browser.textCursor()
        return cursor.position()
    
    def _on_cursor_position_changed(self):
        """Handle cursor position change"""
        cursor = self.browser.textCursor()
        self.current_position = cursor.position()
    
    def _on_selection_changed(self):
        """Handle selection change with debouncing"""
        text = self.get_selected_text()
        if text and len(text.strip()) > 0:
            self._debounce_timer.start(300)  # Debounce for 300ms
    
    def _emit_selection(self):
        """Emit selection signal after debounce"""
        text = self.get_selected_text()
        if text and len(text.strip()) > 0:
            self.text_selected.emit(
                "text_selected",
                self.doc_id,
                text,
                self.current_page,
                self.current_position
            )
    
    def _on_anchor_clicked(self, url: QUrl):
        """Handle anchor click"""
        # Emit signal for external handling
        self.anchor_clicked.emit(url.toString())
    
    def set_html(self, html: str, base_url: QUrl = None):
        """Set HTML content"""
        if base_url:
            self.browser.setHtml(html, base_url)
        else:
            self.browser.setHtml(html)
    
    def find_text(self, text: str, case_sensitive: bool = False) -> bool:
        """Find text in document"""
        flags = QTextDocument.FindFlags()
        if case_sensitive:
            flags |= QTextDocument.FindCaseSensitively
        
        found = self.browser.find(text, flags)
        return found
    
    def find_next(self) -> bool:
        """Find next occurrence"""
        return self.browser.find("")
    
    def find_previous(self) -> bool:
        """Find previous occurrence"""
        return self.browser.find("", QTextDocument.FindBackward)
    
    def set_zoom(self, zoom: float):
        """Set zoom level"""
        # QTextBrowser doesn't support zoom directly
        # We can scale the content using CSS
        if self.content:
            # Inject zoom CSS
            zoom_css = f"body {{ zoom: {zoom}; }}"
            html = self.content.replace('</head>', f'<style>{zoom_css}</style></head>')
            self.browser.setHtml(html)
        self.current_zoom = zoom
    
    anchor_clicked = Signal(str)  # url