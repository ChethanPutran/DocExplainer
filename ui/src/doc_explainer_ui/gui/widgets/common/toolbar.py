from PySide6.QtWidgets import QToolBar, QPushButton, QWidget, QHBoxLayout, QToolButton, QMenu
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

        # Recent files (menu filled by the window each time it opens)
        self.recent_button = QToolButton(self)
        self.recent_button.setText("🕘 Recent")
        self.recent_button.setToolTip("Files you opened recently")
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_menu = QMenu(self.recent_button)
        self.recent_button.setMenu(self.recent_menu)
        self.addWidget(self.recent_button)

        # Library
        self.library_action = QAction("📚 Library", self)
        self.library_action.setStatusTip("All documents on the server: open or delete them")
        self.addAction(self.library_action)

        self.addSeparator()
        
        # View actions
        self.zoom_in_action = QAction("🔍+ Zoom In", self)
        self.zoom_out_action = QAction("🔍- Zoom Out", self)
        self.addAction(self.zoom_in_action)
        self.addAction(self.zoom_out_action)
        
        self.addSeparator()
        
        # Quiz
        self.quiz_action = QAction("\U0001F9E0 Quiz me", self)
        self.quiz_action.setStatusTip("Check your level or practise with a quiz on this document")
        self.addAction(self.quiz_action)

        # Progress dashboard
        self.progress_action = QAction("\U0001F4C8 Progress", self)
        self.progress_action.setStatusTip("Your mastery over time and what to review")
        self.addAction(self.progress_action)

        # Export the conversation
        self.export_action = QAction("\u2B07 Export chat", self)
        self.export_action.setStatusTip("Save this conversation and its notes as Markdown or PDF")
        self.addAction(self.export_action)

        # Job status
        self.jobs_action = QAction("\U0001F4CB Job status", self)
        self.jobs_action.setStatusTip("Show what the server is doing with your documents")
        self.addAction(self.jobs_action)

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