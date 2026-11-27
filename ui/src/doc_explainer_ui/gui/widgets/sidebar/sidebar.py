from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QTabWidget, QVBoxLayout

from ..base.base_widget import BaseWidget
from .chat_panel import ChatPanel
from .feedback_bar import FeedbackBar
from .learn_panel import LearnPanel
from .recommendations_panel import RecommendationsPanel


class Sidebar(BaseWidget):
    """The AI tutor: a conversation (with its sessions and notes), suggested resources, and learning/quizzes."""

    details_requested = Signal()         # "show me the job status"
    feedback_given = Signal(str, list)   # verdict, concept ids the answer was about

    def __init__(self, parent=None, signals=None, speaker=None):
        self._speaker = speaker
        super().__init__(parent, signals)
        self._concept_ids = []

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.tab_widget = QTabWidget()
        self.chat_panel = ChatPanel(speaker=self._speaker)
        self.recommendations_panel = RecommendationsPanel()
        self.learn_panel = LearnPanel()
        self.feedback_bar = FeedbackBar()
        self.tab_widget.addTab(self.chat_panel, "Explanation")
        self.tab_widget.addTab(self.recommendations_panel, "Resources")
        self.tab_widget.addTab(self.learn_panel, "Learn && Quiz")
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.feedback_bar)

        # Shown INSTEAD of the tabs until the document is ready on the server: no AI feature is offered before that.
        self.unavailable_label = QLabel()
        self.unavailable_label.setWordWrap(True)
        self.unavailable_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unavailable_label.setStyleSheet("color: #777; padding: 24px;")
        layout.addWidget(self.unavailable_label)
        self.details_button = QPushButton("View job status and details")
        self.details_button.clicked.connect(self.details_requested.emit)
        layout.addWidget(self.details_button)
        self.setLayout(layout)
        self.set_unavailable("Open a document to use the AI tutor.")

    def _connect_signals(self):
        self.feedback_bar.verdict_given.connect(lambda v: self.feedback_given.emit(v, list(self._concept_ids)))

    # ------------------------------------------------------------------ what a finished answer changes elsewhere
    def on_reply_done(self, result: dict) -> None:
        """The answer is complete: its resources go to the Resources tab, and feedback can now be given about it."""
        explanation = result.get("explanation") or {}
        self._concept_ids = list((explanation.get("context_used") or {}).get("concept_ids", []))
        self.feedback_bar.reset(bool(self._concept_ids))
        self.recommendations_panel.set_resources(explanation.get("resources") or [])

    def show_learn_tab(self):
        self.tab_widget.setCurrentWidget(self.learn_panel)

    def set_unavailable(self, message: str):
        """Hide every AI feature and say why."""
        self.tab_widget.hide()
        self.feedback_bar.hide()
        self.unavailable_label.setText(message)
        self.unavailable_label.show()
        self.details_button.show()

    def set_available(self):
        self.unavailable_label.hide()
        self.details_button.hide()
        self.tab_widget.show()
        self.feedback_bar.show()

    @property
    def available(self) -> bool:
        return not self.unavailable_label.isVisibleTo(self)

    def clear(self):
        """Forget what belongs to the document that was on screen."""
        self.chat_panel.clear()
        self.recommendations_panel.clear()
        self.learn_panel.clear()
        self._concept_ids = []
        self.feedback_bar.reset(False)
