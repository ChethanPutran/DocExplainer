from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QObject, Signal, Slot, QThread

from doc_explainer.orchestrator.progress import (
    CallbackProgressReporter,
)

from ....orchestrator.models import (
    AnswerResponse,
    DocumentResponse,
    ExplainResponse,
    SummarizeResponse,
)

from ....orchestrator.orchestrator import (
    DocExplainerOrchestrator,
)

from ..windows.main_window import MainWindow
from ..windows.about_window import AboutWindow
from ..windows.settings_window import SettingsWindow

from ....config import UIConfig

from ..managers.theme_manager import ThemeManager
from ..managers.shortcut_manager import ShortcutManager

from ..factories.widget_factory import WidgetFactory
from ..utils.signal_utils import SignalInspector
from ..models.signals import UISignals


logger = logging.getLogger(__name__)
# ======================================================================
# DOCUMENT REGISTRATION WORKER
# ======================================================================

class DocumentRegistrationWorker(QObject):
    """
    Executes document registration outside the Qt GUI thread.
    """

    progress = Signal(object)
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        orchestrator: DocExplainerOrchestrator,
        path: str,
        build_graph: bool = True,
        user_id: Optional[str] = None,
    ):
        super().__init__()

        self.orchestrator = orchestrator
        self.path = path
        self.build_graph = build_graph
        self.user_id = user_id

    @Slot()
    def run(self) -> None:
        """
        Execute registration in the worker thread.
        """

        reporter = CallbackProgressReporter(
            self.progress.emit
        )

        try:
            response = self.orchestrator.register_document(
                path=self.path,
                build_graph=self.build_graph,
                user_id=self.user_id,
                progress_reporter=reporter,
            )

            self.finished.emit(response, self.path)

        except Exception as e:
            self.failed.emit(str(e))


# ======================================================================
# WINDOW MANAGER
# ======================================================================

class WindowManager(QObject):
    """Manages windows and application lifecycle."""

    def __init__(
        self,
        config: UIConfig,
        theme_manager: ThemeManager,
        shortcut_manager: ShortcutManager,
        orchestrator: DocExplainerOrchestrator,
        widget_factory: WidgetFactory,
        signal_inspector: Optional[SignalInspector] = None,
    ):
        super().__init__()

        self.config = config
        self.theme_manager = theme_manager
        self.shortcut_manager = shortcut_manager
        self.orchestrator = orchestrator
        self.widget_factory = widget_factory
        self.signal_inspector = signal_inspector

        self.signals = UISignals()
        self.app = QApplication.instance()

        self.main_window: Optional[MainWindow] = None
        self.settings_window: Optional[SettingsWindow] = None
        self.about_window: Optional[AboutWindow] = None

        self.recent_documents: List[str] = []
        self.open_documents: Dict[str, Any] = {}

        # Currently running registration.
        self.registration_thread: Optional[QThread] = None
        self.registration_worker: Optional[
            DocumentRegistrationWorker
        ] = None
        self.registration_active = False

        self._create_main_window()
        self._connect_signals()
        self._load_recent_documents()

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------

    def _create_main_window(self):
        """Create main window."""

        self.main_window = MainWindow(
            window_manager=self,
            config=self.config,
            theme_manager=self.theme_manager,
            shortcut_manager=self.shortcut_manager,
            widget_factory=self.widget_factory,
            signals=self.signals,
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect application signals."""

        self.theme_manager.theme_changed.connect(
            self._on_theme_changed
        )

        self.signals.document_opened.connect(
            self._on_document_opened
        )

        self.signals.document_closed.connect(
            self._on_document_closed
        )

        self.signals.explain_requested.connect(
            self.on_explain
        )

        self.signals.summarize_requested.connect(
            self.on_summarize
        )

        self.signals.ask_requested.connect(
            self.on_ask
        )

        self.signals.follow_up_requested.connect(
            self.on_follow_up
        )

        if self.config.voice.enabled:
            self.signals.voice_input_received.connect(
                self._on_voice_input
            )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _on_theme_changed(self, mode: str):
        """Handle theme changes."""

        logger.info(
            "Theme changed to: %s",
            mode,
        )

        if self.config.theme.mode != mode:
            self.config.theme.mode = mode
            self.config.save()

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    def _on_document_opened(
        self,
        doc_id: str,
        path: str,
    ):
        """Track opened document."""

        logger.info(
            "Document opened: %s with ID %s",
            path,
            doc_id,
        )

        self.open_documents[doc_id] = {
            "path": path,
            "title": Path(path).name,
            "opened_at": datetime.now().isoformat(),
        }

        if path not in self.recent_documents:
            self.recent_documents.insert(
                0,
                path,
            )

            self.recent_documents = (
                self.recent_documents[
                    :self.config.documents.max_recent_files
                ]
            )

            self._save_recent_documents()

    def _on_document_closed(
        self,
        doc_id: str,
    ):
        """Handle document closed."""

        logger.info(
            "Document closed: %s",
            doc_id,
        )

        self.open_documents.pop(
            doc_id,
            None,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on_document_registered(
        self,
        path: str,
        build_graph: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Start asynchronous document registration.

        This method intentionally does NOT return a document ID.
        Registration happens in another thread.
        """

        if user_id is None:
            user_id = self.orchestrator.config.default_user_id

        if build_graph is None:
            build_graph = self.orchestrator.config.backend.knowledge_graph.enabled

        if self.registration_active:
            logger.warning(
                "Document registration already running."
            )

            # if self.main_window:
            #     QMessageBox.warning(
            #         self.main_window,
            #         "Registration in progress",
            #         "Another document is currently being registered.",
            #     )

            return

        logger.info(
            "Starting document registration: %s",
            path,
        )

        # --------------------------------------------------------------
        # Create thread
        # --------------------------------------------------------------

        thread = QThread()

        worker = DocumentRegistrationWorker(
            orchestrator=self.orchestrator,
            path=path,
            build_graph=build_graph,
            user_id=user_id,
        )

        worker.moveToThread(thread)

        # Keep references alive.
        self.registration_thread = thread
        self.registration_worker = worker
        self.registration_active = True

        # --------------------------------------------------------------
        # Thread lifecycle
        # --------------------------------------------------------------

        thread.started.connect(
            worker.run
        )

        # Request shutdown before queued UI completion handlers run.
        worker.finished.connect(
            thread.quit
        )

        worker.failed.connect(
            thread.quit
        )

        worker.progress.connect(
            self._on_registration_progress
        )

        worker.finished.connect(
            self._on_registration_finished
        )

        worker.failed.connect(
            self._on_registration_failed
        )

        # Cleanup worker after thread stops.
        thread.finished.connect(
            worker.deleteLater
        )

        # Cleanup thread.
        thread.finished.connect(
            thread.deleteLater
        )

        # Clear references.
        thread.finished.connect(
            lambda: self._on_registration_thread_finished(thread, worker)
        )

        # --------------------------------------------------------------
        # Start
        # --------------------------------------------------------------

        thread.start()

    # ------------------------------------------------------------------
    # Registration progress
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_registration_progress(
        self,
        event,
    ) -> None:
        """Receive progress from worker."""

        logger.info(
            "[%s] %.0f%% - %s",
            event.step,
            event.progress * 100,
            event.message,
        )

        if self.main_window:
            self.main_window.show_registration_progress(
                event
            )

    # ------------------------------------------------------------------
    # Registration finished
    # ------------------------------------------------------------------

    @Slot(object, str)
    def _on_registration_finished(
        self,
        response,
        path: str,
    ) -> None:
        """Handle successful registration."""

        if not response.success:
            self._on_registration_failed(
                response.message
                if hasattr(response, "message")
                else "Document registration failed."
            )
            return

        if not isinstance(
            response,
            DocumentResponse,
        ):
            self._on_registration_failed(
                "Invalid document registration response."
            )
            return

        if not response.document_id:
            self._on_registration_failed(
                "Document registration returned no document ID."
            )
            return

        doc_id = str(
            response.document_id
        )

        if not path:
            self._on_registration_failed(
                "Unable to determine registered document path."
            )
            return

        logger.info(
            "Document registered successfully: %s",
            doc_id,
        )

        if self.main_window:
            self.main_window.show_registration_complete(
                doc_id
            )

            self.main_window.open_registered_document(
                path=path,
                doc_id=doc_id,
            )

    # ------------------------------------------------------------------
    # Registration failed
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_registration_failed(
        self,
        error: str,
    ) -> None:
        """Handle registration failure."""

        logger.error(
            "Document registration failed: %s",
            error,
        )
        self.registration_active = False

        if self.main_window:
            self.main_window.show_registration_error(
                error
            )

    # ------------------------------------------------------------------
    # Thread cleanup
    # ------------------------------------------------------------------

    @Slot()
    def _on_registration_thread_finished(
        self,
        thread: Optional[QThread] = None,
        worker: Optional[DocumentRegistrationWorker] = None,
    ):
        """Clear worker/thread references."""

        if thread is not None and thread is not self.registration_thread:
            return
        if worker is not None and worker is not self.registration_worker:
            return

        self.registration_worker = None
        self.registration_thread = None
        self.registration_active = False

    def shutdown(self) -> None:
        """Wait for document registration to finish before app shutdown."""
        thread = self.registration_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()

        self.registration_worker = None
        self.registration_thread = None
        self.registration_active = False

    # ------------------------------------------------------------------
    # Voice
    # ------------------------------------------------------------------

    def _on_voice_input(
        self,
        text: str,
    ):
        logger.info(
            "Voice input received: %s",
            text,
        )

    # ------------------------------------------------------------------
    # Recent documents
    # ------------------------------------------------------------------

    def _load_recent_documents(self):
        """Load recent documents."""

        recent_file = (
            Path.home()
            / ".doc_explainer"
            / "recent.json"
        )

        if not recent_file.exists():
            logger.info(
                "No recent documents found."
            )
            return

        try:
            with open(
                recent_file,
                "r",
            ) as f:
                data = json.load(f)

            self.recent_documents = data.get(
                "documents",
                [],
            )

        except Exception as e:
            logger.error(
                "Error loading recent documents: %s",
                e,
            )

    def _save_recent_documents(self):
        """Save recent documents."""

        recent_file = (
            Path.home()
            / ".doc_explainer"
            / "recent.json"
        )

        try:
            recent_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                recent_file,
                "w",
            ) as f:
                json.dump(
                    {
                        "documents": self.recent_documents
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(
                "Error saving recent documents: %s",
                e,
            )

    def get_recent_documents(
        self,
    ) -> List[str]:
        return self.recent_documents

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def on_explain(
        self,
        doc_id: int,
        text: str,
        page: int,
        position: int,
    ):
        """Handle explain action."""

        logger.info(
            "Explain requested for doc %s",
            doc_id,
        )

        section_id = (
            self.orchestrator.get_section_id_at_position(
                str(doc_id),
                page,
                position,
            )
        )

        response = self.orchestrator.explain(
            doc_id=str(doc_id),
            selected_text=text,
            section_id=section_id,
        )

        if response.explanation and response.explanation.context_used.get(
            "error_state",
            False,
        ):
            self._show_ai_error(response.explanation.explanation)
            return

        if (
            response.success
            and isinstance(response, ExplainResponse)
            and response.explanation
        ):
            if self.main_window and self.main_window.sidebar:
                self.main_window.sidebar.update_explanation(
                    response.explanation,
                    section_id,
                )
        else:
            logger.warning(
                "Explain failed for doc %s at section %s",
                doc_id,
                section_id,
            )

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------

    def on_summarize(
        self,
        doc_id: int,
        text: str,
        page: int,
        position: int,
    ):
        """Handle summarize action."""

        logger.info(
            "Summarize requested for doc %s",
            doc_id,
        )

        section_id = (
            self.orchestrator.get_section_id_at_position(
                str(doc_id),
                page,
                position,
            )
        )

        response = self.orchestrator.summarize(
            doc_id=str(doc_id),
            selected_text=text,
            section_id=section_id,
        )

        if response.explanation and response.explanation.context_used.get(
            "error_state",
            False,
        ):
            self._show_ai_error(response.explanation.explanation)
            return

        if (
            response.success
            and isinstance(response, SummarizeResponse)
            and response.explanation
        ):
            if self.main_window and self.main_window.sidebar:
                self.main_window.sidebar.update_explanation(
                    response.explanation,
                    section_id,
                )
        else:
            logger.warning(
                "Summarize failed for doc %s at section %s",
                doc_id,
                section_id,
            )

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------

    def on_ask(
        self,
        doc_id: int,
        text: str,
        page: int,
        position: int,
    ):
        """Handle ask action."""

        logger.info(
            "Ask requested for doc %s",
            doc_id,
        )

        section_id = (
            self.orchestrator.get_section_id_at_position(
                str(doc_id),
                page,
                position,
            )
        )

        response = self.orchestrator.answer(
            doc_id=str(doc_id),
            question=text,
            section_id=section_id,
        )

        if response.explanation and response.explanation.context_used.get(
            "error_state",
            False,
        ):
            self._show_ai_error(response.explanation.explanation)
            return

        if (
            response.success
            and isinstance(response, AnswerResponse)
            and response.explanation
        ):
            if self.main_window and self.main_window.sidebar:
                self.main_window.sidebar.update_explanation(
                    response.explanation,
                    section_id,
                )
        else:
            logger.warning(
                "Ask failed for doc %s at section %s",
                doc_id,
                section_id,
            )

    def _show_ai_error(self, message: str) -> None:
        """Show provider failures to the user without displaying a traceback."""
        if self.main_window:
            QMessageBox.warning(
                self.main_window,
                "AI request failed",
                message,
            )

    # ------------------------------------------------------------------
    # Follow-up
    # ------------------------------------------------------------------

    def on_follow_up(
        self,
        doc_id: str,
        question: str,
        section_id: int,
    ):
        """Handle follow-up question."""

        logger.info(
            "Follow-up question for doc %s: %s",
            doc_id,
            question,
        )

        response = self.orchestrator.answer(
            doc_id=str(doc_id),
            question=question,
            section_id=section_id,
        )

        if (
            response.success
            and isinstance(response, AnswerResponse)
            and response.explanation
        ):
            if self.main_window and self.main_window.sidebar:
                self.main_window.sidebar.update_explanation(
                    response.explanation,
                    section_id,
                )
        else:
            logger.warning(
                "Follow-up failed for doc %s",
                doc_id,
            )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def show_settings(self):
        if not self.settings_window:
            self.settings_window = SettingsWindow(
                self.config,
                self.theme_manager,
                self.main_window,
            )

            self.settings_window.setModal(True)

        self.settings_window.show()
        self.settings_window.raise_()

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def show_about(self):
        if not self.about_window:
            self.about_window = AboutWindow(
                self.main_window
            )

            self.about_window.setModal(True)

        self.about_window.show()
        self.about_window.raise_()

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self):
        if not self.main_window:
            self._create_main_window()

        self.main_window.show()

        if self.config.window.maximized:
            self.main_window.showMaximized()

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def quit(self):
        if self.main_window:
            self.main_window.close()

        if self.settings_window:
            self.settings_window.close()

        if self.about_window:
            self.about_window.close()

        if self.config.window.maximized:
            self.config.window.maximized = True
        else:
            self.config.window.maximized = False

        if self.app:
            self.config.save()
            self.app.quit()