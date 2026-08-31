from PySide6.QtWidgets import QToolBar, QPushButton, QWidget, QHBoxLayout
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QAction

from ..base.base_widget import BaseWidget


class MainToolbar(QToolBar):
    """Main application toolbar"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.setIconSize(QSize(24, 24))
        self._setup_actions()
    
    def _setup_actions(self):
        """Setup toolbar actions"""
        # Open action
        self.open_action = QAction("📂 Open", self)
        self.open_action.setStatusTip("Open document")
        self.addAction(self.open_action)
        
        self.addSeparator()
        
        # View actions
        self.zoom_in_action = QAction("🔍+ Zoom In", self)
        self.zoom_out_action = QAction("🔍- Zoom Out", self)
        self.addAction(self.zoom_in_action)
        self.addAction(self.zoom_out_action)
        
        self.addSeparator()
        
        # Sidebar toggle
        self.toggle_sidebar_action = QAction("📋 Toggle Sidebar", self)
        self.toggle_sidebar_action.setCheckable(True)
        self.toggle_sidebar_action.setChecked(True)
        self.addAction(self.toggle_sidebar_action)
        
        self.addSeparator()
        
        # Theme toggle
        self.theme_action = QAction("🌓 Theme", self)
        self.addAction(self.theme_action)