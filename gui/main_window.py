import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QVBoxLayout, QWidget,
    QPushButton, QToolBar,QSizePolicy,QFileDialog,QHBoxLayout,QTabWidget
)
from PySide6.QtCore import Qt
from .widgets.document_viewer import DocumentViewer
from .widgets.sidebar import Sidebar
from .widgets.voice_input import VoiceInput
from .widgets.voice_output import VoiceOutput
from .utils.utils import load_document  # Utility to read PDF/HTML text
from PySide6.QtWidgets import QFileDialog
from .utils.viewer_factory import create_viewer

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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

        viewer = create_viewer(path)
        viewer.load(path)
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

    def handle_text_action(self, action: str, text: str):
        if action == "EXPLAIN":
            self.sidebar.update_explanation(f"Explanation:\n{text}")
            # Set the text from your sidebar or AI response
            self.voice_output.set_text(text)

        elif action == "SUMMARIZE":
            self.sidebar.update_explanation(f"Summary:\n{text}")
            self.voice_output.set_text(text)
            
        elif action == "ASK":
            self.sidebar.update_explanation(f"Question about:\n{text}")

        elif action == "RELEASE":
            # Optional: preview selection, highlight, etc.
            pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
