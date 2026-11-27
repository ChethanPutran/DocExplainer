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
        self._attach_context_menu(self.browser)
        self._attach_zoom(self.browser)
    
    def _edit_actions(self, menu) -> None:
        copy = menu.addAction("Copy")
        copy.setEnabled(self.browser.textCursor().hasSelection())
        copy.triggered.connect(self.browser.copy)
        menu.addAction("Select all").triggered.connect(self.browser.selectAll)

    def find(self, text: str, forward: bool = True, case_sensitive: bool = False) -> bool:
        from PySide6.QtGui import QTextCursor, QTextDocument
        flags = QTextDocument.FindFlag(0)
        if case_sensitive:
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if not forward:
            flags |= QTextDocument.FindFlag.FindBackward
        if self.browser.find(text, flags):
            return True
        cursor = self.browser.textCursor()                              # wrap around
        cursor.movePosition(QTextCursor.MoveOperation.Start if forward else QTextCursor.MoveOperation.End)
        self.browser.setTextCursor(cursor)
        return self.browser.find(text, flags)

    def highlight_selection(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QTextEdit
        cursor = self.browser.textCursor()
        if not cursor.hasSelection():
            return
        extra = QTextEdit.ExtraSelection()
        extra.cursor = cursor
        extra.format.setBackground(QColor(255, 235, 59, 140))
        self._highlights = getattr(self, "_highlights", []) + [extra]
        self.browser.setExtraSelections(self._highlights)

    def clear_highlights(self) -> None:
        self._highlights = []
        self.browser.setExtraSelections([])

    def section_at_position(self, sections):
        from .section_locator import section_at_position
        return section_at_position(self.browser.document(), self.browser.textCursor().position(), sections)

    def go_to_section(self, page: int, title: str) -> bool:
        """Scroll to the heading: these viewers have no pages, so find its text."""
        document = self.browser.document()
        cursor = document.find(title.strip()[:80]) if title.strip() else None
        if cursor is None or cursor.isNull():
            return False
        self.browser.setTextCursor(cursor)
        self.browser.ensureCursorVisible()
        return True

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
        """Scale the text: QTextBrowser has no CSS zoom, so scale its default font."""
        self.current_zoom = zoom
        base = getattr(self, "_base_point_size", None) or self.browser.font().pointSizeF()
        self._base_point_size = base
        font = self.browser.font()
        font.setPointSizeF(max(1.0, base * zoom))
        self.browser.document().setDefaultFont(font)
        self.browser.setFont(font)

    anchor_clicked = Signal(str)  # url