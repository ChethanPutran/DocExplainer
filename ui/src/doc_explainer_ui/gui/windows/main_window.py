from __future__ import annotations
from typing import TYPE_CHECKING
import os
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QMessageBox, QWidget, QVBoxLayout
)
from ..widgets.common.find_bar import FindBar
from PySide6.QtCore import Qt

from ..factories import WidgetFactory, ViewerFactory
from ..widgets.voice.piper_engine import engine_factory
from ..widgets import BaseDocumentViewer, Sidebar, SentenceSpeaker, VoiceInput, MainToolbar
from ..models.signals import UISignals
from doc_explainer_ui.config import UIConfig

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
        self.tabs.setDocumentMode(True)
        self.find_bar = FindBar()
        centre = QWidget()
        column = QVBoxLayout(centre)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.tabs, 1)
        column.addWidget(self.find_bar)
        self.setCentralWidget(centre)
        self.setAcceptDrops(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)

        # Voice: the microphone and the speaker live in the chat box (like a chat app). The recogniser itself is a
        # hidden widget that owns the microphone; the chat panel's mic button switches it, its words land in the input.
        voice = self.config.voice
        self.speaker = SentenceSpeaker(engine_factory=engine_factory(voice), rate=voice.output_rate, volume=voice.output_volume) if voice.output_enabled else None
        self.voice_input = VoiceInput(self, signals=self.signals, voice_config=voice)
        self.voice_input.hide()

        # Sidebar
        self.sidebar = Sidebar(signals=self.signals, speaker=self.speaker)
        self.dock = QDockWidget("AI Tutor", self)
        self.dock.setWidget(self.sidebar)
        self.dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        chat = self.sidebar.chat_panel
        chat.mic_button.setVisible(voice.enabled)
        chat.speaker_button.setChecked(False)          # reading aloud is opt-in for each session of use

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

        for name, keys, text, how in (("zoom_in", "Ctrl++", "Zoom in", "zoom_in"),
                                      ("zoom_in_eq", "Ctrl+=", "Zoom in", "zoom_in"),
                                      ("zoom_out", "Ctrl+-", "Zoom out", "zoom_out"),
                                      ("zoom_reset", "Ctrl+0", "Reset zoom", "zoom_reset")):
            action = self.shortcut_manager.register_action(name, keys, text, lambda _=False, h=how: self._zoom(h), parent=self)
            self.addAction(action)

        for name, keys, text, slot in (("find", "Ctrl+F", "Find", self._open_find),
                                       ("find_next", "F3", "Find next", lambda: self._find_step(True)),
                                       ("find_prev", "Shift+F3", "Find previous", lambda: self._find_step(False))):
            action = self.shortcut_manager.register_action(name, keys, text, lambda _=False, f=slot: f(), parent=self)
            self.addAction(action)

        # Toggle sidebar shortcut
        self.shortcut_manager.register_action(
            "toggle_sidebar",
            "Ctrl+B",
            "cToggle sidebar",
            self._toggle_sidebar,
            parent=self
        )

    def _zoom(self, how: str):
        viewer = self.tabs.currentWidget()
        if viewer is not None and hasattr(viewer, how):
            getattr(viewer, how)()

    # ------------------------------------------------------------------ find, drag and drop
    def _open_find(self):
        if isinstance(self.tabs.currentWidget(), BaseDocumentViewer):
            self.find_bar.open_bar()

    def _find(self, text: str, forward: bool, case: bool):
        viewer = self.tabs.currentWidget()
        if isinstance(viewer, BaseDocumentViewer):
            self.find_bar.set_result(viewer.find(text, forward, case))

    def _find_step(self, forward: bool):
        if self.find_bar.edit.text():
            self._find(self.find_bar.edit.text(), forward, self.find_bar.case.isChecked())

    def _close_find(self):
        viewer = self.tabs.currentWidget()
        if isinstance(viewer, BaseDocumentViewer):
            viewer.clear_find()
            viewer.setFocus()

    SUPPORTED = (".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pptx")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(self.SUPPORTED) for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(self.SUPPORTED):
                self.open_document(path)
        event.acceptProposedAction()

    def _connect_signals(self):
        """Connect signals"""
        # Toolbar signals
        self.toolbar.open_action.triggered.connect(self._open_document)
        self.toolbar.toggle_sidebar_action.triggered.connect(
            self._toggle_sidebar)
        self.toolbar.zoom_in_action.triggered.connect(lambda: self._zoom("zoom_in"))
        self.toolbar.zoom_out_action.triggered.connect(lambda: self._zoom("zoom_out"))
        self.toolbar.progress_action.triggered.connect(lambda: self._with_doc(self.window_manager.show_progress))
        self.toolbar.export_action.triggered.connect(lambda: self._with_doc(self.window_manager.export_chat))
        self.find_bar.find_requested.connect(self._find)
        self.find_bar.closed.connect(self._close_find)
        self.toolbar.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.toolbar.library_action.triggered.connect(lambda: self.window_manager.show_library())
        self.toolbar.theme_action.triggered.connect(self._toggle_theme)
        self.toolbar.quiz_action.triggered.connect(self._show_quiz)
        self.toolbar.jobs_action.triggered.connect(lambda: self.window_manager.show_jobs())
        self.sidebar.details_requested.connect(lambda: self.window_manager.show_jobs())

        # Chat
        chat = self.sidebar.chat_panel
        wm = self.window_manager
        chat.send_requested.connect(self._ask_from_input)
        chat.stop_requested.connect(lambda: wm.stop_chat())
        chat.new_session_requested.connect(lambda: self._with_doc(wm.new_session))
        chat.session_selected.connect(lambda sid: self._with_doc(wm.select_session, sid))
        chat.rename_session_requested.connect(lambda sid, title: self._with_doc(wm.rename_session, sid, title))
        chat.delete_session_requested.connect(lambda sid: self._with_doc(wm.delete_session, sid))
        chat.note_add_requested.connect(lambda text: self._with_doc(wm.add_note, text))
        chat.note_edit_requested.connect(lambda nid, text: self._with_doc(wm.edit_note, nid, text))
        chat.note_delete_requested.connect(lambda nid: self._with_doc(wm.delete_note, nid))
        chat.summarize_requested.connect(lambda: self._with_doc(wm.summarize_session))
        chat.mic_toggled.connect(self._toggle_microphone)
        self.voice_input.recording_changed.connect(chat.set_listening)
        self.voice_input.status_changed.connect(self._on_voice_status)
        learn, recs = self.sidebar.learn_panel, self.sidebar.recommendations_panel
        learn.diagnostic_requested.connect(lambda level, topics: self._with_doc(self.window_manager.start_diagnostic, level, topics))
        learn.practice_requested.connect(self._practice)
        learn.finished_section.connect(self._finished_section)
        learn.answer_submitted.connect(lambda qid, choice: self._with_doc(self.window_manager.answer_quiz, qid, choice))
        learn.quiz_finished.connect(lambda kind: self._with_doc(self.window_manager.quiz_finished, kind))
        learn.refresh_requested.connect(lambda: self._with_doc(self.window_manager.refresh_learning))
        recs.refresh_requested.connect(lambda: self._with_doc(self.window_manager.refresh_learning))
        recs.section_requested.connect(lambda sid: self._with_doc(self.window_manager.go_to_section, sid))
        recs.concept_requested.connect(lambda name: self._ask_from_input(f"Explain {name}"))
        self.sidebar.feedback_given.connect(lambda verdict, ids: self._with_doc(self.window_manager.send_feedback, verdict, ids))
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Spoken words go into the chat box
        self.voice_input.voice_text.connect(chat.add_transcription)

        # Document signals
        self.signals.text_selected.connect(self._on_text_selected)

    def _current_doc_id(self) -> str:
        current = self.tabs.currentWidget()
        return getattr(current, "doc_id", "") if isinstance(current, BaseDocumentViewer) else ""

    def _with_doc(self, action, *args):
        """Run ``action(doc_id, *args)`` for the document in the active tab (nothing if it is not ready yet)."""
        doc_id = self._current_doc_id()
        if doc_id:
            action(doc_id, *args)
        else:
            self.statusBar().showMessage("Open a document and wait until it is ready first.", 5000)

    def _show_quiz(self):
        """Toolbar \"Quiz me\": open the Learn & Quiz tab (only once the server has the document)."""
        if not self._current_doc_id():
            self.statusBar().showMessage("Quizzes are available once the document has been processed. See the sidebar for progress.", 6000)
            return
        if not self.dock.isVisible():
            self._toggle_sidebar()
        self.sidebar.show_learn_tab()

    def _practice(self):
        self._with_doc(self.window_manager.practice_quiz)

    def _finished_section(self):
        """"I finished this section": the section at the reader's current position in the document."""
        current = self.tabs.currentWidget()
        if not (isinstance(current, BaseDocumentViewer) and current.doc_id):
            self.statusBar().showMessage("Open a document and wait until it is ready first.", 5000)
            return
        self.window_manager.finish_section(current.doc_id, current.get_current_page(), current.get_current_position())

    def _ask_from_input(self, text: str):
        """A question typed (or spoken, or picked as a follow-up): asked at the reader's current position."""
        current = self.tabs.currentWidget()
        if not (isinstance(current, BaseDocumentViewer) and current.doc_id):
            self.statusBar().showMessage("Open a document and wait until it is ready first.", 5000)
            return
        self.window_manager.on_ask(current.doc_id, text, current.get_current_page(), current.get_current_position())

    def _toggle_microphone(self, on: bool):
        if on and not self._current_doc_id():
            self.sidebar.chat_panel.set_listening(False)
            self.statusBar().showMessage("Open a document and wait until it is ready first.", 5000)
            return
        (self.voice_input.start_listening if on else self.voice_input.stop_listening)()

    def _on_voice_status(self, text: str):
        if not self.sidebar.chat_panel._busy:            # a reply's own status ("Writing the reply") takes precedence
            self.sidebar.chat_panel.set_status(text if self.voice_input.is_recording else "")

    def _on_tab_changed(self, index: int):
        if self.voice_input.is_recording:               # never let a question meant for one document reach another
            self.voice_input.stop_listening()
        self.sidebar.clear()
        self._refresh_ai()
        doc_id = self._current_doc_id()
        if doc_id:
            self.window_manager.show_document(doc_id)

    def _refresh_ai(self):
        """Offer the AI features (sidebar tabs, voice questions) only for a document the server has processed.

        Until then the document is fully readable, and the sidebar says what is happening instead.
        """
        viewer = self.tabs.currentWidget()
        if not isinstance(viewer, BaseDocumentViewer):
            self.sidebar.set_unavailable("Open a document to use the AI tutor.")
            ready = False
        elif viewer.doc_id:
            self.sidebar.set_available()
            ready = True
        else:
            note = getattr(viewer, "ai_note", "") or "Document is being processed..."
            self.sidebar.set_unavailable(note + "\n\nYou can read the document now. Explanations, questions and quizzes "
                                                "appear here when it is ready.")
            ready = False
        self.sidebar.chat_panel.set_enabled_for_document(ready)

    def show_section(self, doc_id: str, page: int, title: str = ""):
        """Bring the tab of ``doc_id`` to the front and scroll it to ``page``."""
        for i in range(self.tabs.count()):
            viewer = self.tabs.widget(i)
            if getattr(viewer, "doc_id", None) == doc_id:
                self.tabs.setCurrentIndex(i)
                moved = viewer.go_to_section(page, title) if hasattr(viewer, "go_to_section") else False
                where = f"page {page}" + (f": {title}" if title else "")
                self.statusBar().showMessage(("Jumped to " if moved else "This section is on ") + where, 5000)
                return

    def _fill_recent_menu(self):
        menu = self.toolbar.recent_menu
        menu.clear()
        paths = self.window_manager.get_recent_documents()
        for path in paths:
            action = menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_document(p))
        if not paths:
            menu.addAction("No recent files").setEnabled(False)
        else:
            menu.addSeparator()
            menu.addAction("Clear recent files").triggered.connect(lambda: self.window_manager.clear_recent_documents())

    def viewer_for(self, doc_id: str):
        for i in range(self.tabs.count()):
            if getattr(self.tabs.widget(i), "doc_id", None) == doc_id:
                return self.tabs.widget(i)
        return None

    def focus_document(self, doc_id: str) -> bool:
        """Bring the tab of ``doc_id`` to the front. False if it is not open."""
        for i in range(self.tabs.count()):
            if getattr(self.tabs.widget(i), "doc_id", None) == doc_id:
                self.tabs.setCurrentIndex(i)
                return True
        return False

    def close_document(self, doc_id: str) -> None:
        """Close the tab of ``doc_id`` if it is open (the document was deleted)."""
        for i in range(self.tabs.count()):
            if getattr(self.tabs.widget(i), "doc_id", None) == doc_id:
                self._close_tab(i)
                return

    def _open_document(self):
        """Open document dialog"""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "",
            "All Files (*.*);;PDF Files (*.pdf);;Word (*.docx);;PowerPoint (*.pptx);;Text Files (*.txt *.md);;HTML Files (*.html *.htm)"
        )

        if not path:
            return
        self.open_document(path)

    def _wants_graph(self, path: str) -> bool:
        """The knowledge-graph choice for this file: the setting, or (when it says "ask") the user, once per new file."""
        mode = self.config.knowledge.build_graph
        if mode == "never":
            return False
        if mode == "ask":
            known = getattr(self.window_manager, "is_known_file", lambda p: False)
            if known(path):
                return False                 # already on the server: opening it again changes nothing
            answer = QMessageBox.question(
                self, "Knowledge graph",
                f"Build the knowledge graph for \u201c{os.path.basename(path)}\u201d?\n\n"
                "Yes: explanations adapt to what you know, and quizzes and knowledge tracking work. Takes a few minutes.\n"
                "No: read the document and ask questions only. You can build it later from the Library.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            return answer == QMessageBox.StandardButton.Yes
        return True

    def open_document(self, path: str):
        """Show the document now; register it with the service in the background.

        The file is readable straight away. Asking questions needs the service to have processed it
        (minutes for a long document), so the tab shows a progress marker until then.
        """
        try:
            viewer = ViewerFactory.create_viewer(path, signals=self.signals)
            if not viewer.load(path):
                QMessageBox.warning(self, "Error", f"Failed to load document: {path}")
                return
            name = os.path.basename(path)
            note = getattr(self.window_manager, "note_opened", None)
            if note:
                note(path)
            self.tabs.setCurrentIndex(self.tabs.addTab(viewer, f"\u23f3 {name}"))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open document: {e}")
            return

        build_graph = self._wants_graph(path)

        def retitle(text: str) -> bool:
            try:  # the tab may have been closed while we waited
                index = self.tabs.indexOf(viewer)
                if index < 0:
                    return False
                self.tabs.setTabText(index, text)
                return True
            except RuntimeError:
                return False

        def refresh():
            if self.tabs.currentWidget() is viewer:
                self._refresh_ai()

        def progress(job):
            if job.state == "paused":
                retitle(f"\u23f8 {name}")
                viewer.ai_note = "Document processing is paused. Resume it from Job status to use the AI tutor."
            else:
                retitle(f"\u23f3 {int(job.progress * 100)}% {name}")
                viewer.ai_note = f"Document is being processed: {int(job.progress * 100)}%" + (f"\n{job.stage}" if job.stage else "")
            refresh()

        def ready(doc_id: str):
            if retitle(name):
                viewer.set_doc_id(str(doc_id))
                viewer.ai_note = ""
                self.signals.document_opened.emit(str(doc_id), path)
                refresh()
                if self.tabs.currentWidget() is viewer:
                    self.window_manager.show_document(str(doc_id))

        def failed(message: str):
            if retitle(f"\u26a0 {name}"):
                viewer.ai_note = f"Document could not be processed, so the AI tutor is not available: {message}"
                refresh()
                QMessageBox.warning(self, "Could not process the document", message)

        def cancelled():
            if retitle(f"\u26d4 {name}"):
                viewer.ai_note = "Document processing was cancelled, so the AI tutor is not available for it. Open the document again to retry."
                refresh()

        self.window_manager.register_document_async(path, on_done=ready, on_error=failed, on_progress=progress,
                                                    on_cancelled=cancelled, build_graph=build_graph)

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

    def closeEvent(self, event):
        """Handle window close event"""
        # Close all tabs
        while self.tabs.count():
            self._close_tab(0)

        event.accept()
