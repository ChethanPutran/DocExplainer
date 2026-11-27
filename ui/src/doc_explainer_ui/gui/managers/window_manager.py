from typing import Optional, Dict, Any, List
import logging
import os
import threading
from datetime import datetime 
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget

from doc_explainer_sdk import DocExplainerClient, DocExplainerError, JobCancelledError, JobTimeoutError

from ..windows.main_window import MainWindow
from ..windows.about_window import AboutWindow
from ..windows.jobs_window import JobStatusWindow
from ..windows.library_window import LibraryWindow
from ..windows.progress_window import ProgressWindow
from ..utils.recent_files import RecentFiles
from ..windows.settings_window import SettingsWindow
from doc_explainer_ui.config import UIConfig
from ..managers.theme_manager import ThemeManager
from ..managers.shortcut_manager import ShortcutManager
from ..factories.widget_factory import WidgetFactory
from ..utils.signal_utils import SignalInspector
from ..models.signals import UISignals
from ..utils.background import Background, Cancelled


class WindowManager:
    """Manages windows and application lifecycle"""

    jobs_window: Optional[JobStatusWindow] = None
    _jobs_refreshing = False
    
    def __init__(self, 
                 config: UIConfig,
                 theme_manager: ThemeManager,
                 shortcut_manager: ShortcutManager,
                 client: DocExplainerClient,
                 widget_factory: WidgetFactory,
                 signal_inspector: Optional[SignalInspector] = None):
        
        self.config = config
        self.theme_manager = theme_manager
        self.shortcut_manager = shortcut_manager
        self.client = client
        self.widget_factory = widget_factory
        self.signal_inspector = signal_inspector
        self.signals = UISignals()
        self._bg = Background()
        self._registered: set = set()  # ids the service has finished processing
        self._structures: Dict[str, Any] = {}  # doc_id -> Structure (section id -> page)
        self._sessions: Dict[str, str] = {}    # doc_id -> the conversation session in use
        self._chatting = False
        self._chat_stop: Optional[threading.Event] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app = QApplication.instance()
        self.main_window: Optional[MainWindow] = None
        self.settings_window: Optional[SettingsWindow] = None
        self.about_window: Optional[AboutWindow] = None
        
        self.recents = RecentFiles(Path.home() / '.doc_explainer' / 'recent.json', config.documents.max_recent_files)
        self.library_window: Optional[LibraryWindow] = None
        self.progress_window: Optional[ProgressWindow] = None
        self.open_documents: Dict[str, Any] = {}
        
        self._create_main_window()
        self._connect_signals()
    
    def _create_main_window(self):
        """Create main window"""
        self.main_window = MainWindow(
            window_manager=self,
            config=self.config,
            theme_manager=self.theme_manager,
            shortcut_manager=self.shortcut_manager,
            widget_factory=self.widget_factory,
            signals=self.signals
        )
    
    def _connect_signals(self):
        """Connect signals"""
        # Connect theme changes
        self.theme_manager.theme_changed.connect(self._on_theme_changed)
        
        # Connect document signals
        self.signals.document_opened.connect(self._on_document_opened)
        self.signals.document_closed.connect(self._on_document_closed)

        # Dconnect action signals
        self.signals.explain_requested.connect(self.on_explain)
        self.signals.summarize_requested.connect(self.on_summarize)
        self.signals.ask_requested.connect(self.on_ask)
        self.signals.note_requested.connect(lambda d, text: self.add_note(d, text))
        
        # Connect voice signals if enabled
        if self.config.voice.enabled:
            self.signals.voice_input_received.connect(self._on_voice_input)
    
    def _on_theme_changed(self,  mode: str):
        """Handle theme change"""
        self.logger.info(f"Theme changed to: {mode}")
        self.config.theme.mode = mode

        # Save theme preference
        # Update config only if it's different
        if self.config.theme.mode != mode:
            self.config.theme.mode = mode
            self.config.save()
    
    def _on_document_opened(self, doc_id: str, path: str):
        """Handle document opened"""
        self.logger.info(f"Document opened: {path} with ID: {doc_id}")
        self.open_documents[doc_id] = {
            'path': path,
            'title': Path(path).name,
            'opened_at': datetime.now().isoformat()
        }
        
        self.recents.add(path)
        self.recents.link(doc_id, path)           # lets the Library reopen this document later
    
    def _on_document_closed(self, doc_id: str):
        """Handle document closed"""
        self.logger.info(f"Document closed with ID: {doc_id}")
        if doc_id in self.open_documents:
            del self.open_documents[doc_id]
    
    def _on_voice_input(self, text: str):
        """Handle voice input"""
        self.logger.info(f"Voice input received: {text}")
        # Process voice input through the service client
        # This would trigger the appropriate action
    
    def is_known_file(self, path: str) -> bool:
        """This file was already processed by the server before (so opening it again changes nothing there)."""
        target = os.path.abspath(path)
        return any(os.path.abspath(p) == target for p in self.recents.linked_paths())

    def note_opened(self, path: str) -> None:
        """A file was opened (even before the server has processed it): it goes to the top of the recent list."""
        self.recents.add(path)

    def get_recent_documents(self) -> List[str]:
        """Recently opened files that still exist, newest first."""
        return self.recents.paths()

    def clear_recent_documents(self) -> None:
        self.recents.clear()

    # ------------------------------------------------------------------ progress dashboard, export
    def show_progress(self, doc_id: str) -> None:
        d = str(doc_id)
        if not self._ready(d):
            return
        if self.progress_window is None:
            self.progress_window = ProgressWindow(self.main_window if isinstance(self.main_window, QWidget) else None)
            self.progress_window.practise_requested.connect(lambda: self.practice_quiz(self._progress_doc))
            self.progress_window.refresh_requested.connect(lambda: self.refresh_progress(self._progress_doc))
        self._progress_doc = d
        self.progress_window.show()
        self.progress_window.raise_()
        self.refresh_progress(d)

    def refresh_progress(self, doc_id: str) -> None:
        window = self.progress_window

        def work(emit, cancel):
            doc = self.client.get_document(doc_id)
            if doc.knowledge not in ("ready", "partial"):
                return None, [], []
            return (self.client.get_learner(doc_id).model_dump(mode="json"),
                    [h.model_dump(mode="json") for h in self.client.get_history(doc_id)],
                    [r.model_dump(mode="json") for r in self.client.get_review(doc_id)])

        self._bg.run(work, on_done=lambda r: window.set_data(*r),
                     on_error=lambda e: window.set_error(getattr(e, "message", None) or str(e)))

    @staticmethod
    def write_export(path: str, markdown: str) -> None:
        """Save a conversation as Markdown, or as PDF when the file name ends in .pdf."""
        if path.lower().endswith(".pdf"):
            from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument
            writer = QPdfWriter(path)
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            doc = QTextDocument()
            doc.setMarkdown(markdown)
            doc.print_(writer)
        else:
            Path(path).write_text(markdown, encoding="utf-8")

    def export_chat(self, doc_id: str) -> None:
        d = str(doc_id)
        session = self._sessions.get(d)
        if not session:
            self._status("There is no conversation to export yet.", 5000)
            return

        def done(export):
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self.main_window, "Export conversation", export.filename,
                                                  "Markdown (*.md);;PDF (*.pdf)")
            if not path:
                return
            try:
                self.write_export(path, export.markdown)
                self._status(f"Saved {os.path.basename(path)}", 5000)
            except OSError as e:
                self._notify_error("Saving the conversation", e)

        self._bg.run(lambda emit, cancel: self.client.export_session(d, session), on_done=done,
                     on_error=lambda e: self._notify_error("Exporting the conversation", e))

    # ------------------------------------------------------------------ library
    def show_library(self) -> None:
        """Open (or raise) the library window and fetch the documents now."""
        if self.library_window is None:
            self.library_window = LibraryWindow(self.main_window if isinstance(self.main_window, QWidget) else None)
            self.library_window.refresh_requested.connect(self.refresh_library)
            self.library_window.open_requested.connect(self.open_from_library)
            self.library_window.delete_requested.connect(self.delete_from_library)
            self.library_window.build_graph_requested.connect(self.build_graph_for)
            self.library_window.remove_graph_requested.connect(self.remove_graph_for)
        self.library_window.show()
        self.library_window.raise_()
        self.refresh_library()

    def refresh_library(self) -> None:
        window = self.library_window
        if window is None:
            return
        self._bg.run(lambda emit, cancel: self.client.list_documents(),
                     on_done=lambda docs: window.set_documents([d.model_dump(mode="json") for d in docs]),
                     on_error=lambda e: window.set_error(getattr(e, "message", None) or str(e)))

    def open_from_library(self, doc_id: str) -> None:
        """Show the tab if the document is already open; otherwise open the local file it came from (asking for it if unknown)."""
        if self.main_window.focus_document(doc_id):
            return
        path = self.recents.path_for(doc_id)
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self.library_window, "Where is this file on your computer?", "",
                "Documents (*.pdf *.txt *.md *.html *.htm *.docx *.pptx);;All Files (*.*)")
            if not path:
                return
        self.main_window.open_document(path)

    def build_graph_for(self, doc_id: str) -> None:
        """Build the knowledge graph of a document that has none (or an incomplete one): same document, no copy."""
        path = self.recents.path_for(doc_id)
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self.library_window, "Where is this file on your computer?", "",
                "Documents (*.pdf *.txt *.md *.html *.htm *.docx *.pptx);;All Files (*.*)")
            if not path:
                return
        self._status("Building the knowledge graph (this can take several minutes)...")

        def done(new_id):
            self._registered.add(str(new_id))
            self.recents.link(new_id, path)
            self._status("Knowledge graph ready", 5000)
            self.refresh_library()
            if self.main_window.focus_document(str(new_id)):
                self.refresh_learning(str(new_id))

        self.register_document_async(path, on_done=done, build_graph=True,
                                     on_error=lambda m: (self._status(f"Knowledge graph failed: {m}", 10000), self.refresh_library()))

    def remove_graph_for(self, doc_id: str) -> None:
        def done(_):
            self._status("Knowledge graph removed", 5000)
            self.refresh_library()
            if self.main_window.focus_document(doc_id):
                self.refresh_learning(doc_id)

        self._bg.run(lambda emit, cancel: self.client.delete_knowledge(doc_id), on_done=done,
                     on_error=lambda e: (self._notify_error("Removing the knowledge graph", e), self.refresh_library()))

    def delete_from_library(self, doc_id: str) -> None:
        def done(_):
            self._registered.discard(doc_id)
            self._structures.pop(doc_id, None)
            self._sessions.pop(doc_id, None)
            self.recents.forget_document(doc_id)
            self.main_window.close_document(doc_id)
            self._status("Document deleted from the library", 5000)
            self.refresh_library()

        def failed(err):
            self._notify_error("Deleting the document", err)
            self.refresh_library()

        self._bg.run(lambda emit, cancel: self.client.delete_document(doc_id), on_done=done, on_error=failed)

    # ------------------------------------------------------------------ service calls
    # Every call to the service runs on a worker thread (Background); results come back here on the
    # GUI thread. Registration in particular can take many minutes (it is a background job on the
    # server), so nothing in this class may block on it.

    def _status(self, message: str, ms: int = 0) -> None:
        status_bar = getattr(self.main_window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message, ms)

    def _notify_error(self, action: str, err: Exception) -> None:
        """Log a failed service call and tell the user in the status bar (never raise into Qt)."""
        message = getattr(err, "message", None) or str(err)
        self.logger.warning(f"{action} failed: {err}")
        self._status(f"{action} failed: {message}", 10000)

    @staticmethod
    def _describe(job) -> str:
        if job.state == "queued":
            where = f" (position {job.queue_position})" if job.queue_position else ""
            return f"Waiting to process the document{where}..."
        if job.state == "running":
            return "Processing the document (this can take several minutes)..."
        if job.state == "paused":
            return "Document processing is paused. Resume it from Job status."
        if job.state == "cancelled":
            return "Document processing was cancelled"
        return "Document ready" if job.state == "succeeded" else "Document processing failed"

    def register_document_async(self, path: str, on_done=None, on_error=None, on_progress=None, on_cancelled=None,
                                build_graph: bool = True) -> None:
        """Upload ``path`` and wait for the service to process it, without blocking the UI.

        ``on_done(doc_id)`` / ``on_error(message)`` / ``on_progress(job)`` run on the GUI thread.
        Uploading identical content again is instant: the service returns the existing document.
        """
        self.logger.info(f"Registering document: {path}")

        def work(emit, cancel):
            job = self.client.start_registration(path, build_graph=build_graph)
            emit(job)
            while not job.done:
                if cancel.is_set():
                    raise Cancelled()
                try:  # wait in short slices so a closing application can interrupt us
                    job = self.client.wait_for_job(job.job_id, timeout=2.0, poll_interval=0.5, on_update=emit)
                except JobTimeoutError as t:
                    job = t.job or job
            if job.state == "failed":  # re-raise through the SDK's own mapping
                self.client.wait_for_job(job.job_id)
            return job.document

        def done(document):
            self._registered.add(document.doc_id)
            self._structures.pop(document.doc_id, None)
            self.logger.info(f"Document registered with ID: {document.doc_id}")
            self._status("Document ready", 4000)
            if on_done:
                on_done(document.doc_id)

        def failed(err):
            if isinstance(err, JobCancelledError):          # the user's own choice: not an error to apologise for
                self._status("Document processing was cancelled", 6000)
                if on_cancelled:
                    on_cancelled()
                elif on_error:
                    on_error("Processing was cancelled.")
                return
            self._notify_error("Opening the document", err)
            if on_error:
                on_error(getattr(err, "message", None) or str(err))

        def progress(job):
            self._status(self._describe(job))
            if on_progress:
                on_progress(job)

        self._bg.run(work, on_done=done, on_error=failed, on_progress=progress)

    def register_in_background(self, path: str) -> None:
        """Fire-and-forget registration (recent documents, command line)."""
        self.register_document_async(path)

    def _ready(self, doc_id) -> bool:
        if str(doc_id) in self._registered:
            return True
        self._status("This document is still being processed. Try again when it is ready.", 6000)
        return False

    # ------------------------------------------------------------------ the conversation
    def _chat_panel(self):
        return self.main_window.sidebar.chat_panel

    def _chat(self, doc_id, kind: str, text: str, locate, display: Optional[str] = None) -> None:
        """One turn, streamed: the question appears at once, then status messages, then the reply as it is written."""
        d = str(doc_id)
        if not self._ready(d):
            return
        if self._chatting:
            self._status("The tutor is still answering. Stop it first, or wait a moment.", 4000)
            return
        panel = self._chat_panel()
        stop = self._chat_stop = threading.Event()
        self._chatting = True
        panel.add_user_message(display or text)
        panel.begin_reply()
        panel.set_status("Understanding your request")
        session_id = self._sessions.get(d)

        def work(emit, cancel):
            section = locate()
            events = self.client.chat_stream(d, text, kind=kind, section_id=section, session_id=session_id)
            try:
                for ev in events:
                    if stop.is_set() or cancel.is_set():
                        raise Cancelled()
                    emit(ev)
            finally:
                events.close()                      # drops the connection if we stopped early

        def progress(ev):
            if stop.is_set():
                return                              # the reader pressed Stop: ignore what is still in flight
            if ev.type == "session":
                self._sessions[d] = ev.data["id"]
            elif ev.type == "status":
                panel.set_status(ev.data.get("text", ""))
            elif ev.type == "delta":
                panel.append_delta(ev.text)
            elif ev.type == "done":
                panel.finish_reply(ev.data)
                self.main_window.sidebar.on_reply_done(ev.data)

        def done(_):
            if stop is self._chat_stop:
                self._chatting = False
            if not stop.is_set():
                self.load_sessions(d, show=False)

        def failed(err):
            if stop is self._chat_stop:
                self._chatting = False
            if stop.is_set():
                return
            self._notify_error(kind.capitalize(), err)
            panel.fail_reply(getattr(err, "message", None) or str(err))

        self._bg.run(work, on_done=done, on_error=failed, on_progress=progress)

    def stop_chat(self) -> None:
        """The reader pressed Stop: keep what has been written so far."""
        if self._chat_stop is not None:
            self._chat_stop.set()
        self._chatting = False
        self._chat_panel().stop_reply()

    def _locator(self, d: str, page: int, position: int):
        """The section a question is about. Viewers without pages find it from the text position; others use the page."""
        def locate():
            viewer = getattr(self.main_window, "viewer_for", lambda _d: None)(d)
            find = getattr(viewer, "section_at_position", None)
            if find is not None:
                structure = self._structures.get(d) or self.client.get_structure(d)
                self._structures[d] = structure
                section = find([(int(s.ordinal), s.title) for s in structure.sections])
                if section is not None:
                    return section
            return self.client.get_section_id(d, page, position)
        return locate

    @staticmethod
    def _quote(text: str, n: int = 160) -> str:
        text = " ".join(text.split())
        return text if len(text) <= n else text[:n].rstrip() + "..."

    def on_explain(self, doc_id, text: str, page: int, position: int):
        """Handle explain action"""
        self.logger.info(f"Explain requested for doc {doc_id}")
        d = str(doc_id)
        self._chat(d, "explain", text, self._locator(d, page, position), f"Explain: \u201c{self._quote(text)}\u201d")

    def on_summarize(self, doc_id, text: str, page: int, position: int):
        """Handle summarize action"""
        self.logger.info(f"Summarize requested for doc {doc_id}")
        d = str(doc_id)
        self._chat(d, "summarize", text, self._locator(d, page, position), f"Summarize: \u201c{self._quote(text)}\u201d")

    def on_ask(self, doc_id, text: str, page: int, position: int):
        """Handle ask action (typed, spoken, a follow-up, or from the right-click menu)"""
        self.logger.info(f"Ask requested for doc {doc_id}")
        d = str(doc_id)
        self._chat(d, "ask", text, self._locator(d, page, position))

    # ------------------------------------------------------------------ sessions and notes
    def load_sessions(self, doc_id: str, select: Optional[str] = None, show: bool = True) -> None:
        """Fetch the document's sessions; ``show`` also displays the chosen one's history."""
        d = str(doc_id)

        def work(emit, cancel):
            sessions = self.client.list_sessions(d)
            chosen = select or self._sessions.get(d) or (sessions[0].id if sessions else "")
            if chosen and not any(s.id == chosen for s in sessions):
                chosen = sessions[0].id if sessions else ""
            detail = self.client.get_session(d, chosen) if (show and chosen) else None
            return sessions, chosen, detail

        def done(result):
            sessions, chosen, detail = result
            if chosen:
                self._sessions[d] = chosen
            else:
                self._sessions.pop(d, None)
            if not self._is_current(d):
                return
            panel = self._chat_panel()
            panel.set_sessions([s.model_dump() for s in sessions], chosen)
            if show:
                if detail is not None:
                    panel.show_history([m.model_dump() for m in detail.messages], [n.model_dump() for n in detail.notes])
                else:
                    panel.show_history([], [])

        self._bg.run(work, on_done=done, on_error=lambda e: self._notify_error("Loading conversations", e))

    def _session_action(self, d: str, label: str, call, then=None) -> None:
        if not self._ready(d):
            return

        def done(result):
            if then:
                then(result)

        self._bg.run(lambda emit, cancel: call(), on_done=done, on_error=lambda e: self._notify_error(label, e))

    def select_session(self, doc_id: str, session_id: str) -> None:
        d = str(doc_id)
        if self._chatting:
            self._status("Wait for the reply to finish before switching sessions.", 4000)
            self.load_sessions(d, show=False)
            return
        self._sessions[d] = session_id
        self.load_sessions(d, select=session_id)

    def new_session(self, doc_id: str) -> None:
        d = str(doc_id)
        self._session_action(d, "Creating a session", lambda: self.client.create_session(d),
                             lambda s: (self._sessions.__setitem__(d, s.id), self.load_sessions(d, select=s.id)))

    def rename_session(self, doc_id: str, session_id: str, title: str) -> None:
        d = str(doc_id)
        self._session_action(d, "Renaming the session", lambda: self.client.rename_session(d, session_id, title),
                             lambda _: self.load_sessions(d, show=False))

    def delete_session(self, doc_id: str, session_id: str) -> None:
        d = str(doc_id)

        def after(_):
            if self._sessions.get(d) == session_id:
                self._sessions.pop(d, None)
            self.load_sessions(d)
        self._session_action(d, "Deleting the session", lambda: self.client.delete_session(d, session_id), after)

    def _ensure_session(self, d: str) -> str:
        """The session notes go to: the one in use, else a new one (called on a worker thread)."""
        sid = self._sessions.get(d)
        if sid:
            return sid
        sid = self.client.create_session(d).id
        self._sessions[d] = sid
        return sid

    def _refresh_notes(self, d: str) -> None:
        sid = self._sessions.get(d)
        if not sid:
            return

        def done(detail):
            if self._is_current(d):
                self._chat_panel().set_notes([n.model_dump() for n in detail.notes])
                self.load_sessions(d, show=False)
        self._bg.run(lambda emit, cancel: self.client.get_session(d, sid), on_done=done,
                     on_error=lambda e: self._notify_error("Loading notes", e))

    def add_note(self, doc_id: str, text: str) -> None:
        d = str(doc_id)
        self._session_action(d, "Saving the note", lambda: self.client.add_note(d, self._ensure_session(d), text),
                             lambda _: (self._status("Note saved", 3000), self._refresh_notes(d)))

    def edit_note(self, doc_id: str, note_id: str, text: str) -> None:
        d = str(doc_id)
        self._session_action(d, "Saving the note", lambda: self.client.edit_note(d, self._ensure_session(d), note_id, text),
                             lambda _: self._refresh_notes(d))

    def delete_note(self, doc_id: str, note_id: str) -> None:
        d = str(doc_id)
        self._session_action(d, "Deleting the note", lambda: self.client.delete_note(d, self._ensure_session(d), note_id),
                             lambda _: self._refresh_notes(d))

    def summarize_session(self, doc_id: str) -> None:
        d = str(doc_id)
        if not self._sessions.get(d):
            self._status("Ask something first: there is no conversation to summarize yet.", 5000)
            return
        sid = self._sessions[d]
        self._status("Writing a summary of this session...")
        self._session_action(d, "Summarizing the session", lambda: self.client.summarize_session(d, sid),
                             lambda _: (self._status("Summary added to your notes", 5000), self._refresh_notes(d)))

    def finish_section(self, doc_id: str, page: int, position: int) -> None:
        """"I finished this section": the section the reader is looking at right now."""
        d = str(doc_id)

        def work(emit, cancel):
            section = self.client.get_section_id(d, page, position)
            self.client.mark_progress(d, section)
            return section

        if not self._ready(d):
            return
        self._bg.run(work, on_done=lambda s: (self._status("Noted: you finished this section", 4000), self.refresh_learning(d)),
                     on_error=lambda e: self._notify_error("Saving your progress", e))

    # ------------------------------------------------------------------ learning: knowledge, quizzes, feedback, resources
    # Results are dropped when the user has meanwhile switched to another document.

    def _learn(self):
        return self.main_window.sidebar.learn_panel

    def _is_current(self, doc_id) -> bool:
        return self._learn().doc_id == str(doc_id)

    def show_document(self, doc_id: str) -> None:
        """A document became the active one: fill the Learn and Resources tabs for it."""
        d = str(doc_id)
        sidebar = self.main_window.sidebar
        sidebar.learn_panel.set_document(d)
        sidebar.recommendations_panel.clear()
        sidebar.recommendations_panel.set_document(d)
        if d in self._registered:
            self.refresh_learning(d)
            self.load_sessions(d)

    def refresh_learning(self, doc_id: str) -> None:
        """Knowledge-build status, the learner's level and what to review."""
        d = str(doc_id)

        def work(emit, cancel):
            doc = self.client.get_document(d)
            learner = None
            if doc.knowledge in ("ready", "partial"):
                learner = self.client.get_learner(d)
            return doc, learner

        def done(result):
            doc, learner = result
            if not self._is_current(d):
                return
            panel = self._learn()
            panel.set_knowledge(doc.knowledge, doc.sections, doc.concepts)
            if learner:
                beliefs = [b.model_dump() for b in learner.concepts]
                panel.set_learner(learner.level, beliefs)
                self.main_window.sidebar.recommendations_panel.set_review(beliefs)

        self._bg.run(work, on_done=done, on_error=lambda e: self._notify_error("Loading learning status", e))

    def start_diagnostic(self, doc_id: str, level: str, topics: list) -> None:
        d = str(doc_id)
        self._quiz_call(d, "diagnostic", lambda: self.client.start_diagnostic(d, level, known_topics=topics).questions)

    def practice_quiz(self, doc_id: str, section_id: Optional[int] = None) -> None:
        d = str(doc_id)
        self._quiz_call(d, "practice", lambda: self.client.practice_quiz(d, section_id=section_id, count=5))

    def _quiz_call(self, d: str, kind: str, call) -> None:
        if not self._ready(d):
            return
        self._status("Preparing questions...")

        def done(questions):
            self._status("Questions ready", 3000)
            if self._is_current(d):
                self._learn().show_questions([q.model_dump() for q in questions], kind)

        def failed(e):
            self._notify_error("Preparing questions", e)
            if self._is_current(d):
                self._learn().show_questions([], kind)
                self._learn().set_message(getattr(e, "message", None) or str(e))

        self._bg.run(lambda emit, cancel: call(), on_done=done, on_error=failed)

    def answer_quiz(self, doc_id: str, question_id: str, choice: int) -> None:
        d = str(doc_id)

        def done(result):
            if self._is_current(d):
                self._learn().show_answer(result.model_dump(), choice)

        def failed(e):
            self._notify_error("Checking the answer", e)
            if self._is_current(d):
                self._learn().show_answer_error(getattr(e, "message", None) or str(e))

        self._bg.run(lambda emit, cancel: self.client.answer_quiz(d, question_id, choice), on_done=done, on_error=failed)

    def quiz_finished(self, doc_id: str, kind: str) -> None:
        """Finishing a level check re-estimates the learner; either way, refresh what the tutor knows."""
        d = str(doc_id)
        if kind != "diagnostic":
            self.refresh_learning(d)
            return

        def done(_):
            self._status("Your level was updated", 4000)
            self.refresh_learning(d)

        self._bg.run(lambda emit, cancel: self.client.finish_diagnostic(d), on_done=done,
                     on_error=lambda e: self._notify_error("Updating your level", e))

    def send_feedback(self, doc_id: str, verdict: str, concept_ids: list) -> None:
        d = str(doc_id)
        if not concept_ids:
            return

        def done(_):
            self._status("Thanks: later explanations will take that into account", 4000)
            self.refresh_learning(d)

        self._bg.run(lambda emit, cancel: self.client.send_feedback(d, verdict, concept_ids), on_done=done,
                     on_error=lambda e: self._notify_error("Sending feedback", e))

    def mark_progress(self, doc_id: str, section_id: int) -> None:
        d = str(doc_id)
        if d in self._registered:
            self._bg.run(lambda emit, cancel: self.client.mark_progress(d, section_id),
                         on_error=lambda e: self._notify_error("Saving your progress", e))

    def go_to_section(self, doc_id: str, section_id: str) -> None:
        """Scroll the document to a section named by a ``doc://`` resource."""
        d = str(doc_id)

        def work(emit, cancel):
            structure = self._structures.get(d) or self.client.get_structure(d)
            self._structures[d] = structure
            return next((s for s in structure.sections if s.id == section_id), None)

        def done(section):
            if section is None:
                self._status("That section is no longer in the document", 5000)
            else:
                self.main_window.show_section(d, section.page_start, section.title)

        self._bg.run(work, on_done=done, on_error=lambda e: self._notify_error("Opening the section", e))

    def show_jobs(self) -> None:
        """Open (or raise) the job status window and fetch fresh data now; it refreshes itself while visible."""
        if self.jobs_window is None:
            self.jobs_window = JobStatusWindow(self.main_window if isinstance(self.main_window, QWidget) else None)
            self.jobs_window.refresh_requested.connect(self.refresh_jobs)
            self.jobs_window.pause_requested.connect(self.pause_job)
            self.jobs_window.resume_requested.connect(self.resume_job)
            self.jobs_window.cancel_requested.connect(self.cancel_job)
        self.jobs_window.show()
        self.jobs_window.raise_()
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        window = self.jobs_window
        if window is None or self._jobs_refreshing:
            return
        self._jobs_refreshing = True          # a slow server must not pile up requests behind the timer

        def finish():
            self._jobs_refreshing = False

        def done(jobs):
            finish()
            window.set_jobs([j.model_dump(mode="json") for j in jobs])

        def failed(err):
            finish()
            window.set_error(getattr(err, "message", None) or str(err))

        self._bg.run(lambda emit, cancel: self.client.list_jobs(), on_done=done, on_error=failed)

    def _job_action(self, label: str, call) -> None:
        def done(_):
            self.refresh_jobs()
        self._bg.run(lambda emit, cancel: call(), on_done=done, on_error=lambda e: (self._notify_error(label, e), self.refresh_jobs()))

    def pause_job(self, job_id: str) -> None:
        self._job_action("Pausing", lambda: self.client.pause_job(job_id))

    def resume_job(self, job_id: str) -> None:
        self._job_action("Resuming", lambda: self.client.resume_job(job_id))

    def cancel_job(self, job_id: str) -> None:
        self._job_action("Cancelling", lambda: self.client.cancel_job(job_id))

    def show_settings(self):
        """Show settings window"""
        if not self.settings_window:
            self.settings_window = SettingsWindow(
                self.config,
                self.theme_manager,
                self.main_window
            )
            self.settings_window.setModal(True)
        
        self.settings_window.show()
        self.settings_window.raise_()
    
    def show_about(self):
        """Show about window"""
        if not self.about_window:
            self.about_window = AboutWindow(self.main_window)
            self.about_window.setModal(True)
        
        self.about_window.show()
        self.about_window.raise_()
    
    def launch(self):
        """Launch the application"""
        self.main_window.show()
        
        # Apply window state from config
        if self.config.window.maximized:
            self.main_window.showMaximized()
    
    def quit(self):
        """Quit the application"""
        # Stop background work (daemon threads: exit never waits on them)
        self._bg.shutdown()

        # Save configuration
        self.config.save()
        
        # Close all windows
        if self.main_window:
            self.main_window.close()
        
        if self.settings_window:
            self.settings_window.close()
        
        if self.about_window:
            self.about_window.close()

        if self.jobs_window:
            self.jobs_window.close()

        if self.library_window:
            self.library_window.close()

        if self.progress_window:
            self.progress_window.close()
        
        # Quit application
        self.app.quit()


