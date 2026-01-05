from enum import Enum

import fitz

class ActionType(Enum):
    EXPLAIN = "EXPLAIN"
    SUMMARIZE = "SUMMARIZE"
    ASK = "ASK"
    RELEASE = "RELEASE"
    SELECT = "SELECT"


class BaseViewer:
    """
    Interface-like base class (NO ABC, NO QObject)
    """
    doc : fitz.Document 

    def load(self, path: str):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def get_document(self) -> fitz.Document | None:
        return self.doc
