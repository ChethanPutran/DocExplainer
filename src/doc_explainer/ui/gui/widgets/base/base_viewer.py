from abc import abstractmethod
from PySide6.QtCore import Signal
from .base_widget import BaseWidget

class BaseDocumentViewer(BaseWidget):
    """Base class for document viewers"""
    doc_id: str
    # Signal emitted when text is selected
    text_selected = Signal(str, str, int, int)  # action, doc_id, text, page, position
    # Signal emitted when document is loaded
    document_loaded = Signal(str)  # doc_id
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.current_path: str = ""
        self.current_page: int = 1
        self.current_position: int = 0
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load document from path"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear document and free resources"""
        pass
    
    def get_selected_text(self) -> str:
        """Get currently selected text - to be overridden"""
        return ""
    
    def get_current_page(self) -> int:
        """Get current page number"""
        return self.current_page
    
    def get_current_position(self) -> int:
        """Get current text position"""
        return self.current_position
    
    def set_doc_id(self, doc_id: str):
        """Set document ID"""
        self.doc_id = doc_id
    
    def emit_text_action(self, action: str, text: str):
        """Emit text selection signal"""
        self.text_selected.emit(
            action,
            self.doc_id,
            text,
            self.current_page,
            self.current_position
        )