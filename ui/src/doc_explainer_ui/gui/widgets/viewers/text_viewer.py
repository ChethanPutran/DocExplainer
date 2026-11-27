from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QTextCursor, QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QTextDocument

from .document_viewer import DocumentViewer
from ...utils.file_utils import FileUtils


class TextViewer(DocumentViewer):
    """Text document viewer"""
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.content = ""
        self.highlighter = None
    
    def _setup_ui(self):
        """Setup text viewer UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # Set monospace font
        font = QFont("Courier New", 10)
        self.text_edit.setFont(font)
        
        # Connect signals
        self.text_edit.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.text_edit.selectionChanged.connect(self._on_selection_changed)
        
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self._attach_context_menu(self.text_edit)
        self._attach_zoom(self.text_edit)
    
    def _edit_actions(self, menu) -> None:
        copy = menu.addAction("Copy")
        copy.setEnabled(self.text_edit.textCursor().hasSelection())
        copy.triggered.connect(self.text_edit.copy)
        menu.addAction("Select all").triggered.connect(self.text_edit.selectAll)

    def find(self, text: str, forward: bool = True, case_sensitive: bool = False) -> bool:
        from PySide6.QtGui import QTextCursor, QTextDocument
        flags = QTextDocument.FindFlag(0)
        if case_sensitive:
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        if not forward:
            flags |= QTextDocument.FindFlag.FindBackward
        if self.text_edit.find(text, flags):
            return True
        cursor = self.text_edit.textCursor()                              # wrap around
        cursor.movePosition(QTextCursor.MoveOperation.Start if forward else QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        return self.text_edit.find(text, flags)

    def highlight_selection(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QTextEdit
        cursor = self.text_edit.textCursor()
        if not cursor.hasSelection():
            return
        extra = QTextEdit.ExtraSelection()
        extra.cursor = cursor
        extra.format.setBackground(QColor(255, 235, 59, 140))
        self._highlights = getattr(self, "_highlights", []) + [extra]
        self.text_edit.setExtraSelections(self._highlights)

    def clear_highlights(self) -> None:
        self._highlights = []
        self.text_edit.setExtraSelections([])

    def section_at_position(self, sections):
        from .section_locator import section_at_position
        return section_at_position(self.text_edit.document(), self.text_edit.textCursor().position(), sections)

    def go_to_section(self, page: int, title: str) -> bool:
        """Scroll to the heading: these viewers have no pages, so find its text."""
        document = self.text_edit.document()
        cursor = document.find(title.strip()[:80]) if title.strip() else None
        if cursor is None or cursor.isNull():
            return False
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
        return True

    def load(self, path: str) -> bool:
        """Load text document"""
        try:
            self.content = FileUtils.read_text_file(path)
            if self.content is None:
                return False
            
            self.text_edit.setText(self.content)
            self.current_path = path
            
            # Set syntax highlighting based on extension
            ext = FileUtils.get_file_extension(path)
            self._setup_highlighter(ext)
            
            self.document_loaded.emit(self.doc_id)
            return True
            
        except Exception as e:
            print(f"Error loading text file: {e}")
            return False
    
    def _setup_highlighter(self, extension: str):
        """Setup syntax highlighter based on file extension"""
        # This would set up appropriate syntax highlighting
        # For now, just use a simple highlighter
        pass
    
    def clear(self) -> bool:
        """Clear document"""
        self.text_edit.clear()
        self.content = ""
        self.current_path = ""
        self.current_page = 1
        self.current_position = 0
        return True
    
    def get_selected_text(self) -> str:
        """Get selected text"""
        cursor = self.text_edit.textCursor()
        return cursor.selectedText()
    
    def get_current_position(self) -> int:
        """Get current text position"""
        cursor = self.text_edit.textCursor()
        return cursor.position()
    
    def _on_cursor_position_changed(self):
        """Handle cursor position change"""
        cursor = self.text_edit.textCursor()
        self.current_position = cursor.position()
        
        # Update status bar if available
        parent = self.parent()
        while parent:
            if hasattr(parent, 'status_bar'):
                parent.status_bar.set_position_info(
                    self.current_position,
                    len(self.content)
                )
                break
            parent = parent.parent()
    
    def _on_selection_changed(self):
        """Handle selection change"""
        text = self.get_selected_text()
        if text and len(text.strip()) > 0:
            # Emit text selected signal
            self.text_selected.emit(
                "text_selected",
                self.doc_id,
                text,
                self.current_page,
                self.current_position
            )
    
    def find_text(self, text: str, case_sensitive: bool = False) -> bool:
        """Find text in document"""
        flags = QTextDocument.FindFlags()
        if case_sensitive:
            flags |= QTextDocument.FindCaseSensitively
        
        found = self.text_edit.find(text, flags)
        return found
    
    def find_next(self) -> bool:
        """Find next occurrence"""
        return self.text_edit.find("")
    
    def find_previous(self) -> bool:
        """Find previous occurrence"""
        return self.text_edit.find("", QTextDocument.FindBackward)
    
    def set_zoom(self, zoom: float):
        """Set zoom level"""
        font = self.text_edit.font()
        font.setPointSize(int(10 * zoom))
        self.text_edit.setFont(font)
        self.current_zoom = zoom