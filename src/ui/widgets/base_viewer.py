from enum import Enum
from abc import ABC, abstractmethod
import fitz

class ActionType(Enum):
    EXPLAIN = "EXPLAIN"
    SUMMARIZE = "SUMMARIZE"
    ASK = "ASK"
    RELEASE = "RELEASE"
    SELECT = "SELECT"


class BaseViewer:
    """
    Interface-like base class for document viewers.
    """
    doc : fitz.Document 
    doc_id : str | None

    @abstractmethod
    def load(self, path: str):
        pass
    @abstractmethod
    def clear(self):
        pass
    
    def get_document(self) -> fitz.Document | None:
        return self.doc
   
    def set_doc_id(self, doc_id: str):
        self.doc_id = doc_id