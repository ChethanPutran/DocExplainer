from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QLabel,
)

from PySide6.QtCore import Qt

from doc_explainer.orchestrator.progress import (
    ProgressEvent,
    ProgressStatus,
)

from ..factories import WidgetFactory, ViewerFactory
from ..widgets import (
    BaseDocumentViewer,
    Sidebar,
    VoiceInput,
    VoiceOutput,
    MainToolbar,
)

from ..models.signals import UISignals

from ....config import UIConfig

if TYPE_CHECKING:
    from ..managers import (
        ShortcutManager,
        ThemeManager,
    )


class MainWindow(QMainWindow):
    """Main application window."""

    sidebar: Sidebar
    toolbar: MainToolbar

    def __init__(
        self,
        window_manager,
        theme_manager: ThemeManager,
        shortcut_manager: ShortcutManager,
        widget_factory: WidgetFactory,
        signals: UISignals,
        config: UIConfig,
    ):
        super().__init__()

        self.window_manager = window_manager
        self.theme_manager = theme_manager
        self.shortcut_manager = shortcut_manager
        self.widget_factory = widget_factory
        self.signals = signals
        self.config = config

        self.setWindowTitle(
            "Doc Explainer"
        )

        self.resize(
            1200,
            800,
        )

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

    # ==================================================================
    # UI
    # ==================================================================

    def _setup_ui(self):
        """Setup main window UI."""

        # --------------------------------------------------------------
        # Document tabs
        # --------------------------------------------------------------

        self.tabs = QTabWidget()

        self.tabs.setTabsClosable(
            True
        )

        self.tabs.tabCloseRequested.connect(
            self._close_tab
        )

        self.setCentralWidget(
            self.tabs
        )

        # --------------------------------------------------------------
        # Sidebar
        # --------------------------------------------------------------

        self.sidebar = Sidebar(
            signals=self.signals
        )

        self.dock = QDockWidget(
            "AI Tutor",
            self,
        )

        self.dock.setWidget(
            self.sidebar
        )

        self.dock.setFloating(
            False
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.dock,
        )

        # --------------------------------------------------------------
        # Voice controls
        # --------------------------------------------------------------

        voice_container = QWidget()

        voice_layout = QVBoxLayout()

        voice_layout.setContentsMargins(
            5,
            5,
            5,
            5,
        )

        self.voice_input = VoiceInput(
            signals=self.signals
        )

        self.voice_output = VoiceOutput(
            signals=self.signals
        )

        voice_layout.addWidget(
            self.voice_input
        )

        voice_layout.addWidget(
            self.voice_output
        )

        voice_container.setLayout(
            voice_layout
        )

        self.dock.setTitleBarWidget(
            voice_container
        )

        # --------------------------------------------------------------
        # Toolbar
        # --------------------------------------------------------------

        self.toolbar = MainToolbar()

        self.addToolBar(
            self.toolbar
        )

        # --------------------------------------------------------------
        # Registration progress
        # --------------------------------------------------------------

        self.registration_container = QWidget()

        registration_layout = QVBoxLayout(
            self.registration_container
        )

        registration_layout.setContentsMargins(
            8,
            4,
            8,
            4,
        )

        self.registration_status = QLabel(
            "Ready"
        )

        self.registration_progress = QProgressBar()

        self.registration_progress.setRange(
            0,
            100,
        )

        self.registration_progress.setValue(
            0
        )

        registration_layout.addWidget(
            self.registration_status
        )

        registration_layout.addWidget(
            self.registration_progress
        )

        self.registration_container.hide()

        self.statusBar().addPermanentWidget(
            self.registration_container
        )

    # ==================================================================
    # Shortcuts
    # ==================================================================

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""

        self.shortcut_manager.register_action(
            "close_tab",
            "Ctrl+W",
            "Close current tab",
            self._close_current_tab,
            parent=self,
        )

        self.shortcut_manager.register_action(
            "open_document",
            "Ctrl+O",
            "Open document",
            self._open_document,
            parent=self,
        )

        self.shortcut_manager.register_action(
            "toggle_sidebar",
            "Ctrl+B",
            "Toggle sidebar",
            self._toggle_sidebar,
            parent=self,
        )

    # ==================================================================
    # Signals
    # ==================================================================

    def _connect_signals(self):
        """Connect signals."""

        self.toolbar.open_action.triggered.connect(
            self._open_document
        )

        self.toolbar.toggle_sidebar_action.triggered.connect(
            self._toggle_sidebar
        )

        self.toolbar.theme_action.triggered.connect(
            self._toggle_theme
        )

        self.sidebar.question_clicked.connect(
            self._on_follow_up_clicked
        )

        self.voice_input.voice_text.connect(
            self._on_voice_input
        )

        self.signals.text_selected.connect(
            self._on_text_selected
        )

    # ==================================================================
    # Open document
    # ==================================================================

    def _open_document(self):
        """
        Ask the user for a document and start asynchronous registration.
        """

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            "",
            (
                "All Files (*.*);;"
                "PDF Files (*.pdf);;"
                "Text Files (*.txt);;"
                "HTML Files (*.html)"
            ),
        )

        if not path:
            return

        # IMPORTANT:
        #
        # Do not wait for a document ID here.
        #
        # Registration happens in QThread.
        self.window_manager.on_document_registered(
            path
        )

    # ==================================================================
    # Registration progress
    # ==================================================================

    def show_registration_progress(
        self,
        event: ProgressEvent,
    ):
        """Update registration progress UI."""

        self.registration_container.show()

        self.registration_status.setText(
            event.message
        )

        self.registration_progress.setValue(
            int(event.progress * 100)
        )

        if event.status == ProgressStatus.FAILED:
            self.registration_progress.setValue(
                0
            )

    # ==================================================================
    # Registration complete
    # ==================================================================

    def show_registration_complete(
        self,
        doc_id: str,
    ):
        """Show registration completion."""

        self.registration_status.setText(
            f"Document registered: {doc_id}"
        )

        self.registration_progress.setValue(
            100
        )

        # Hide after successful completion.
        self.registration_container.hide()

    # ==================================================================
    # Registration error
    # ==================================================================

    def show_registration_error(
        self,
        error: str,
    ):
        """Show registration error."""

        self.registration_status.setText(
            "Document registration failed."
        )

        self.registration_progress.setValue(
            0
        )

        self.registration_container.hide()

        QMessageBox.critical(
            self,
            "Document Registration Failed",
            error,
        )

    # ==================================================================
    # Open registered document
    # ==================================================================

    def open_registered_document(
        self,
        path: str,
        doc_id: str,
    ):
        """
        Create and display the document viewer after backend
        registration has completed successfully.
        """

        try:
            viewer = ViewerFactory.create_viewer(
                path,
                signals=self.signals,
            )

            viewer.set_doc_id(
                str(doc_id)
            )

            if not viewer.load(path):
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load document: {path}",
                )
                return

            name = os.path.basename(
                path
            )

            index = self.tabs.addTab(
                viewer,
                name,
            )

            self.tabs.setCurrentIndex(
                index
            )

            self.signals.document_opened.emit(
                str(doc_id),
                path,
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open document: {e}",
            )

    # ==================================================================
    # Close tab
    # ==================================================================

    def _close_current_tab(self):
        """Close currently active tab."""

        index = self.tabs.currentIndex()

        if index >= 0:
            self._close_tab(
                index
            )

    def _close_tab(
        self,
        index: int,
    ):
        """Close tab at index."""

        widget = self.tabs.widget(
            index
        )

        if not widget:
            return

        doc_id = getattr(
            widget,
            "doc_id",
            None,
        )

        try:
            if hasattr(
                widget,
                "clear",
            ):
                widget.clear()

        except Exception:
            pass

        self.tabs.removeTab(
            index
        )

        widget.deleteLater()

        if doc_id:
            self.signals.document_closed.emit(
                str(doc_id)
            )

    # ==================================================================
    # Sidebar
    # ==================================================================

    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""

        if self.dock.isVisible():
            self.dock.hide()

            self.toolbar.toggle_sidebar_action.setChecked(
                False
            )

        else:
            self.dock.show()

            self.toolbar.toggle_sidebar_action.setChecked(
                True
            )

        self.signals.sidebar_toggled.emit(
            self.dock.isVisible()
        )

    # ==================================================================
    # Theme
    # ==================================================================

    def _toggle_theme(self):
        """Toggle theme."""

        self.theme_manager.toggle_theme()

    # ==================================================================
    # Text selection
    # ==================================================================

    def _on_text_selected(
        self,
        doc_id: str,
        text: str,
        page: int,
        position: int,
    ):
        """Handle text selection."""

        print(
            f"Text selected: {text[:50]}..."
        )

    # ==================================================================
    # Follow-up
    # ==================================================================

    def _on_follow_up_clicked(
        self,
        question: str,
        section_id: int,
    ):
        """Handle follow-up question."""

        current = self.tabs.currentWidget()

        if not current:
            return

        if isinstance(
            current,
            BaseDocumentViewer,
        ):
            doc_id = current.doc_id

            self.window_manager.on_follow_up(
                doc_id,
                question,
                section_id,
            )

    # ==================================================================
    # Voice
    # ==================================================================

    def _on_voice_input(
        self,
        text: str,
    ):
        """Handle voice input."""

        current = self.tabs.currentWidget()

        if not current:
            return

        if isinstance(
            current,
            BaseDocumentViewer,
        ):
            self.window_manager.on_ask(
                doc_id=int(current.doc_id),
                text=text,
                page=current.get_current_page(),
                position=current.get_current_position(),
            )

    # ==================================================================
    # Close application
    # ==================================================================

    def closeEvent(self, event):
        """Handle window close."""

        while self.tabs.count():
            self._close_tab(0)

        event.accept()