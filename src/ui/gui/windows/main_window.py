import os
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout,
    QApplication, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QAction

from ..widgets.sidebar.sidebar import Sidebar
from ..widgets.voice.voice_input import VoiceInput
from ..widgets.voice.voice_output import VoiceOutput
from ..widgets.common.toolbar import MainToolbar
from ..factories.viewer_factory import ViewerFactory
from ..models.signals import UISignals
from ..config import UIConfig


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, window_manager, config: UIConfig = None):
        super().__init__()
        self.window_manager = window_manager
        self.config = config or UIConfig()
        self.signals = UISignals()
        
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
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        
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
        close_action = QAction(self)
        close_action.setShortcut(QKeySequence("Ctrl+W"))
        close_action.triggered.connect(lambda: self._close_tab(self.tabs.currentIndex()))
        self.addAction(close_action)
        
        # Open document shortcut
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_document)
        self.addAction(open_action)
        
        # Toggle sidebar shortcut
        toggle_action = QAction(self)
        toggle_action.setShortcut(QKeySequence("Ctrl+B"))
        toggle_action.triggered.connect(self._toggle_sidebar)
        self.addAction(toggle_action)
    
    def _connect_signals(self):
        """Connect signals"""
        # Toolbar signals
        self.toolbar.open_action.triggered.connect(self._open_document)
        self.toolbar.toggle_sidebar_action.triggered.connect(self._toggle_sidebar)
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
                
                # Emit signal
                self.signals.document_opened.emit(str(doc_id), path)
            else:
                QMessageBox.warning(self, "Error", f"Failed to load document: {path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open document: {e}")
    
    def _close_tab(self, index: int):
        """Close tab at given index"""
        widget = self.tabs.widget(index)
        if widget:
            doc_id = getattr(widget, 'doc_id', None)
            
            # Clear viewer resources
            if hasattr(widget, 'clear'):
                widget.clear()
            
            # Remove tab
            self.tabs.removeTab(index)
            
            # Schedule deletion
            widget.deleteLater()
            
            # Emit signal
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
        pass
    
    def _on_text_selected(self, doc_id: str, text: str, page: int, position: int):
        """Handle text selection from any viewer"""
        # This would trigger explanation based on the selected text
        # For now, just log it
        print(f"Text selected: {text[:50]}...")
    
    def _on_follow_up_clicked(self, question: str, section_id: int):
        """Handle follow-up question click"""
        current = self.tabs.currentWidget()
        if not current or not hasattr(current, 'doc_id'):
            return
        
        doc_id = current.doc_id
        self.window_manager.on_follow_up(doc_id, question, section_id)
    
    def _on_voice_input(self, text: str):
        """Handle voice input"""
        print(f"Voice input: {text}")
        
        # Treat as a question
        current = self.tabs.currentWidget()
        if current and hasattr(current, 'doc_id'):
            self.window_manager.on_ask(
                doc_id=int(current.doc_id),
                text=text,
                page=current.get_current_page(),
                position=current.get_current_position()
            )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Close all tabs
        while self.tabs.count():
            self._close_tab(0)
        
        event.accept()