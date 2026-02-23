from enum import Enum
from abc import abstractmethod
import fitz
from PySide6.QtCore import Signal

class ActionType(Enum):
    EXPLAIN = "EXPLAIN"
    SUMMARIZE = "SUMMARIZE"
    ASK = "ASK"
    RELEASE = "RELEASE"
    SELECT = "SELECT"

class BaseViewer:
    """
    Interface-like base class for document viewers.
    Ensures consistent signal signatures across different document types.
    """
    # action, doc_id, text, page, position
    text_action = Signal(str, str, str, int, int)

    def __init__(self):
        super().__init__()
        self.doc_id = None
        self.doc_model = None
        self.last_selection = ""
        self.doc = None

    def set_doc_id(self, doc_id: str):
        self.doc_id = doc_id

    def set_model(self, model):
        self.doc_model = model

    def get_document(self) -> fitz.Document | None:
        return self.doc

    def get_selected_text(self) -> str:
        return self.last_selection
    
    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def clear(self):
        pass