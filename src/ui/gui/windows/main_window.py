from __future__ import annotations
from typing import TYPE_CHECKING
import os
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout,  QMessageBox
)
from PySide6.QtCore import Qt

from ..factories import WidgetFactory, ViewerFactory
from ..widgets import BaseDocumentViewer, Sidebar,  VoiceInput, VoiceOutput, MainToolbar
from ..models.signals import UISignals
from src.config import UIConfig

if TYPE_CHECKING:
    from ..managers import ShortcutManager, ThemeManager


class MainWindow(QMainWindow):
    """Main application window"""
    sidebar: Sidebar
    toolbar: MainToolbar

    def __init__(self, window_manager,
                 theme_manager: ThemeManager,
                 shortcut_manager: ShortcutManager,
                 widget_factory: WidgetFactory,
                 signals: UISignals,
                 config: UIConfig):
        super().__init__()
        self.window_manager = window_manager
        self.theme_manager = theme_manager
        self.shortcut_manager = shortcut_manager
        self.widget_factory = widget_factory
        self.signals = signals
        self.config = config

        self.setWindowTitle("Doc Explainer")
        self.resize(1200, 800)

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

    def _setup_ui(self):
        """Setup main window UI"""
        # Tab widget for documents
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.setCentralWidget(self.tabs)
        self.tabs.tabCloseRequested.connect(self._close_tab)

        # Sidebar
        self.sidebar = Sidebar(signals=self.signals)
        self.dock = QDockWidget("AI Tutor", self)
        self.dock.setWidget(self.sidebar)
        self.dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        # Voice controls
        voice_container = QWidget()
        voice_layout = QVBoxLayout()
        voice_layout.setContentsMargins(5, 5, 5, 5)

        self.voice_input = VoiceInput(signals=self.signals)
        self.voice_output = VoiceOutput(signals=self.signals)

        voice_layout.addWidget(self.voice_input)
        voice_layout.addWidget(self.voice_output)
        voice_container.setLayout(voice_layout)

        self.dock.setTitleBarWidget(voice_container)

        # Toolbar
        self.toolbar = MainToolbar()
        self.addToolBar(self.toolbar)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Close tab shortcut
        self.shortcut_manager.register_action(
            "close_tab",
            "Ctrl+W",
            "cClose current tab",
            self._close_tab,
            parent=self
        )
        # Open document shortcut
        self.shortcut_manager.register_action(
            "open_document",
            "Ctrl+O",
            "cOpen document",
            self._open_document,
            parent=self
        )

        # Toggle sidebar shortcut
        self.shortcut_manager.register_action(
            "toggle_sidebar",
            "Ctrl+B",
            "cToggle sidebar",
            self._toggle_sidebar,
            parent=self
        )

    def _connect_signals(self):
        """Connect signals"""
        # Toolbar signals
        self.toolbar.open_action.triggered.connect(self._open_document)
        self.toolbar.toggle_sidebar_action.triggered.connect(
            self._toggle_sidebar)
        self.toolbar.theme_action.triggered.connect(self._toggle_theme)

        # Sidebar signals
        self.sidebar.question_clicked.connect(self._on_follow_up_clicked)

        # Voice signals
        self.voice_input.voice_text.connect(self._on_voice_input)

        # Document signals
        self.signals.text_selected.connect(self._on_text_selected)

    def _open_document(self):
        """Open document dialog"""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "",
            "All Files (*.*);;PDF Files (*.pdf);;Text Files (*.txt);;HTML Files (*.html)"
        )

        if not path:
            return

        try:
            # Register document with backend
            doc_id = self.window_manager.on_document_registered(path)

            # Create viewer
            viewer = ViewerFactory.create_viewer(path, signals=self.signals)

            # Load document
            if viewer.load(path):
                viewer.set_doc_id(str(doc_id))

                # Add to tabs
                name = os.path.basename(path)
                index = self.tabs.addTab(viewer, name)
                self.tabs.setCurrentIndex(index)

                # Emit signal to notify document opened
                self.signals.document_opened.emit(str(doc_id), path)
            else:
                QMessageBox.warning(
                    self, "Error", f"Failed to load document: {path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to open document: {e}")

    def _close_tab(self, index: int):
        """Close tab at given index"""
        widget = self.tabs.widget(index)
        if widget:
            doc_id = getattr(widget, 'doc_id', None)

            # Clear viewer resources
            if hasattr(widget, 'clear'):
                widget.close()  # Close any open resources
                widget.clear()  # Clear loaded content

            # Remove tab
            self.tabs.removeTab(index)

            # Schedule deletion
            widget.deleteLater()

            # Emit signal to notify document closed
            if doc_id:
                self.signals.document_closed.emit(doc_id)

    def _toggle_sidebar(self):
        """Toggle sidebar visibility"""
        if self.dock.isVisible():
            self.dock.hide()
            self.toolbar.toggle_sidebar_action.setChecked(False)
        else:
            self.dock.show()
            self.toolbar.toggle_sidebar_action.setChecked(True)

        self.signals.sidebar_toggled.emit(self.dock.isVisible())

    def _toggle_theme(self):
        """Toggle theme"""
        # This would change the application stylesheet
        self.theme_manager.toggle_theme()

    def _on_text_selected(self, doc_id: str, text: str, page: int, position: int):
        """Handle text selection from any viewer"""
        # This would trigger explanation based on the selected text
        # For now, just log it
        print(f"Text selected: {text[:50]}...")

    def _on_follow_up_clicked(self, question: str, section_id: int):
        """Handle follow-up question click"""
        current = self.tabs.currentWidget()
        if not current:
            return
        if isinstance(current, BaseDocumentViewer):
            doc_id = current.doc_id
            self.window_manager.on_follow_up(doc_id, question, section_id)
        else:
            print("No active document to send follow-up question to.")

    def _on_voice_input(self, text: str):
        """Handle voice input"""
        print(f"Voice input: {text}")

        current = self.tabs.currentWidget()
        if not current:
            return
        if isinstance(current, BaseDocumentViewer):
            self.window_manager.on_ask(
                doc_id=int(current.doc_id),
                text=text,
                page=current.get_current_page(),
                position=current.get_current_position()
            )
        else:
            print("No active document to send voice input to.")

    def closeEvent(self, event):
        """Handle window close event"""
        # Close all tabs
        while self.tabs.count():
            self._close_tab(0)

        event.accept()
