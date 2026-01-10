import os
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QVBoxLayout, QWidget,
    QPushButton, QToolBar,QSizePolicy,QFileDialog,QHBoxLayout,QTabWidget
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from src.api.actions import APIActions
from src.ui.widgets.sidebar import Sidebar
from src.ui.widgets.voice_input import VoiceInput
from src.ui.widgets.voice_output import VoiceOutput
from src.ui.utils.utils import load_document  # Utility to read PDF/HTML text
from src.ui.utils.viewer_factory import create_viewer


class MainWindow(QMainWindow):
    # explain_requested = Signal(str)  # text to explain
    # summarize_requested = Signal(str)  # text to summarize
    # ask_requested = Signal(str)  # question to ask
    def __init__(self,window_manager):
        super().__init__()
        self.window_manager = window_manager
        self.setWindowTitle("Doc Explainer")

        # # Central Widget: Document Viewer
        # self.doc_viewer = DocumentViewer()
        # self.setCentralWidget(self.doc_viewer)

        # Tab widget for documents
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.setCentralWidget(self.tabs)
        self.tabs.tabCloseRequested.connect(self.close_tab)


        # Sidebar Dock: Explanation, Recommendations
        self.sidebar = Sidebar()
        self.dock = QDockWidget("AI Tutor", self)
        self.dock.setWidget(self.sidebar)
        self.dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # Voice Input & Output
        self.voice_input = VoiceInput()
        self.voice_input = VoiceInput()
        self.voice_input.voice_text.connect(self.handle_voice_text)


        self.voice_output = VoiceOutput()

        # Optional: connect signals
        self.voice_output.tts_started.connect(lambda: print("TTS started"))
        self.voice_output.tts_finished.connect(lambda: print("TTS finished"))


        # Layout for voice buttons inside sidebar
        voice_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.voice_input)
        layout.addWidget(self.voice_output)
        voice_widget.setLayout(layout)
        self.dock.setTitleBarWidget(voice_widget)

        # Toolbar with toggle button
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Container for Open/Close Buttons
        self.file_buttons_container = QWidget()
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)  # No padding
        self.file_buttons_container.setLayout(h_layout)

        # Open Button
        self.open_file_btn = QPushButton("Open Document")
        self.open_file_btn.clicked.connect(self.open_document)
        h_layout.addWidget(self.open_file_btn)

        toolbar.addWidget(self.file_buttons_container)


        # Spacer to push button to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        toggle_sidebar_btn = QPushButton("Toggle Sidebar")
        toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        toolbar.addWidget(toggle_sidebar_btn)

        
        # Connect document selection signal to sidebar update
        # self.doc_viewer.text_selected.connect(self.handle_text_selection)

    def handle_voice_text(self, text):
            print("Voice input received:", text)
            # You can send `text` to your AI sidebar or LLM

    def close_tab(self, index):
        """Close tab at given index."""
        self.tabs.removeTab(index)

    def open_document(self):
        """Open a file dialog to select and load a document."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            "",
            "All Files (*.*)"
        )
        if not path:
            return

        doc_id = self.window_manager.on_document_registered(path)

        viewer = create_viewer(path)

        viewer.set_doc_id(doc_id)  

        viewer.text_action.connect(self.handle_text_action)

        name = os.path.basename(path)
        self.tabs.addTab(viewer, name)
        self.tabs.setCurrentWidget(viewer)

    def toggle_sidebar(self):
        """Show/hide the sidebar dock."""
        if self.dock.isVisible():
            self.dock.hide()
        else:
            self.dock.show()

    def get_selection(self) -> str:
        """Retrieve selected text from the current document viewer."""
        current_viewer = self.tabs.currentWidget()
        if current_viewer:
            return current_viewer.get_selected_text()
        return ""
    
    def handle_text_action(self, action: str, doc_id: str, text: str):
        section_id = self.get_user_selection()
        if action == APIActions.EXPLAIN:
            self.window_manager.on_explain(
                doc_id=doc_id,
                text=text,
                section_id=section_id
            )

        elif action == APIActions.SUMMARIZE:
            self.window_manager.on_summarize(
                doc_id=doc_id,
                text=text,
                section_id=section_id
            )

        elif action == APIActions.ASK:
            self.window_manager.on_ask(
                doc_id=doc_id,
                text=text,
                section_id=section_id
            )

        elif action == APIActions.RELEASE:
            # Optional: preview selection, highlight, etc.
            pass


