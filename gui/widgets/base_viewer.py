from enum import Enum

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

    def load(self, path: str):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
