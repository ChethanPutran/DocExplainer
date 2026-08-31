from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, QObject
from ...models.signals import UISignals


class BaseWidget(QWidget):
    """Base class for all widgets"""
    
    def __init__(self, parent=None, signals: UISignals = None):
        super().__init__(parent)
        self.signals = signals or UISignals()
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup widget UI - to be overridden"""
        pass
    
    def _connect_signals(self):
        """Connect signals - to be overridden"""
        pass
    
    def update_theme(self, theme: str):
        """Update widget theme"""
        pass