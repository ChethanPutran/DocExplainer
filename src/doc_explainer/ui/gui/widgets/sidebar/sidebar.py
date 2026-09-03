from PySide6.QtWidgets import QVBoxLayout, QWidget, QTabWidget
from PySide6.QtCore import Signal

from .explanation_panel import ExplanationPanel
from .recommendations_panel import RecommendationsPanel
from .follow_up_panel import FollowUpPanel
from ..base.base_widget import BaseWidget


class Sidebar(BaseWidget):
    """Main sidebar widget with tabs for different panels"""
    
    # Signal emitted when a follow-up question is clicked
    question_clicked = Signal(str, int)  # question, section_id
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.current_section_id = 0
    
    def _setup_ui(self):
        """Setup sidebar UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create panels
        self.explanation_panel = ExplanationPanel()
        self.recommendations_panel = RecommendationsPanel()
        self.follow_up_panel = FollowUpPanel()
        
        # Add panels to tabs
        self.tab_widget.addTab(self.explanation_panel, "Explanation")
        self.tab_widget.addTab(self.recommendations_panel, "Resources")
        self.tab_widget.addTab(self.follow_up_panel, "Follow-up")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect signals"""
        self.follow_up_panel.question_clicked.connect(self._on_question_clicked)
    
    def _on_question_clicked(self, question: str):
        """Handle follow-up question click"""
        self.question_clicked.emit(question, self.current_section_id)
    
    def update_explanation(self, explanation, section_id: int):
        """Update explanation display"""
        self.current_section_id = section_id
        
        # Update explanation panel
        self.explanation_panel.set_explanation(
            explanation.explanation,
            explanation.known_concepts_used,
            explanation.unknown_concepts_explained
        )
        
        # Update follow-up panel
        self.follow_up_panel.set_questions(explanation.follow_up_questions)
        
        # Update recommendations panel
        resources = getattr(explanation, "resources", None) or getattr(
            explanation,
            "suggested_resources",
            [],
        )
        self.recommendations_panel.set_resources(resources)
        
        # Switch to explanation tab
        self.tab_widget.setCurrentIndex(0)
    
    def clear(self):
        """Clear all panels"""
        self.explanation_panel.clear()
        self.recommendations_panel.clear()
        self.follow_up_panel.clear()