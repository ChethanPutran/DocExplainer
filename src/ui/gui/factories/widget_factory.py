from typing import Optional, Dict, Any
from PySide6.QtWidgets import QWidget

from ..widgets.sidebar.sidebar import Sidebar
from ..widgets.sidebar.explanation_panel import ExplanationPanel
from ..widgets.sidebar.recommendations_panel import RecommendationsPanel
from ..widgets.sidebar.follow_up_panel import FollowUpPanel
from ..widgets.voice.voice_input import VoiceInput
from ..widgets.voice.voice_output import VoiceOutput
from ..widgets.common.toolbar import MainToolbar
from ..widgets.common.status_bar import StatusBar
from ..models.signals import UISignals


class WidgetFactory:
    """Factory for creating UI widgets"""
    
    def __init__(self, signals: Optional[UISignals] = None):
        self.signals = signals or UISignals()
        self._widgets: Dict[str, QWidget] = {}
    
    def create_sidebar(self, parent: Optional[QWidget] = None) -> Sidebar:
        """Create sidebar widget"""
        key = 'sidebar'
        if key not in self._widgets:
            self._widgets[key] = Sidebar(parent, self.signals)
        return self._widgets[key]
    
    def create_explanation_panel(self, parent: Optional[QWidget] = None) -> ExplanationPanel:
        """Create explanation panel"""
        key = 'explanation_panel'
        if key not in self._widgets:
            self._widgets[key] = ExplanationPanel(parent)
        return self._widgets[key]
    
    def create_recommendations_panel(self, parent: Optional[QWidget] = None) -> RecommendationsPanel:
        """Create recommendations panel"""
        key = 'recommendations_panel'
        if key not in self._widgets:
            self._widgets[key] = RecommendationsPanel(parent)
        return self._widgets[key]
    
    def create_follow_up_panel(self, parent: Optional[QWidget] = None) -> FollowUpPanel:
        """Create follow-up panel"""
        key = 'follow_up_panel'
        if key not in self._widgets:
            self._widgets[key] = FollowUpPanel(parent)
        return self._widgets[key]
    
    def create_voice_input(self, parent: Optional[QWidget] = None) -> VoiceInput:
        """Create voice input widget"""
        key = 'voice_input'
        if key not in self._widgets:
            self._widgets[key] = VoiceInput(parent, self.signals)
        return self._widgets[key]
    
    def create_voice_output(self, parent: Optional[QWidget] = None) -> VoiceOutput:
        """Create voice output widget"""
        key = 'voice_output'
        if key not in self._widgets:
            self._widgets[key] = VoiceOutput(parent, self.signals)
        return self._widgets[key]
    
    def create_toolbar(self, parent: Optional[QWidget] = None) -> MainToolbar:
        """Create main toolbar"""
        key = 'toolbar'
        if key not in self._widgets:
            self._widgets[key] = MainToolbar(parent)
        return self._widgets[key]
    
    def create_status_bar(self, parent: Optional[QWidget] = None) -> StatusBar:
        """Create status bar"""
        key = 'status_bar'
        if key not in self._widgets:
            self._widgets[key] = StatusBar(parent)
        return self._widgets[key]
    
    def get_widget(self, key: str) -> Optional[QWidget]:
        """Get widget by key"""
        return self._widgets.get(key)
    
    def clear(self):
        """Clear all widgets"""
        self._widgets.clear()