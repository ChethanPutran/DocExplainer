from typing import Optional, Dict, Any, List
import logging
from datetime import datetime 
from pathlib import Path
from PySide6.QtWidgets import QApplication

from ....orchestrator.models import AnswerResponse, DocumentResponse, ExplainResponse, SummarizeResponse
from ....orchestrator.orchestrator import DocExplainerOrchestrator

from ..windows.main_window import MainWindow
from ..windows.about_window import AboutWindow
from ..windows.settings_window import SettingsWindow
from ..config import UIConfig
from ..managers.theme_manager import ThemeManager
from ..managers.shortcut_manager import ShortcutManager
from ..factories.widget_factory import WidgetFactory
from ..utils.signal_utils import SignalInspector
from ..models.signals import UISignals


class WindowManager:
    """Manages windows and application lifecycle"""
    
    def __init__(self, 
                 config: UIConfig,
                 theme_manager: ThemeManager,
                 shortcut_manager: ShortcutManager,
                 orchestrator: DocExplainerOrchestrator,
                 widget_factory: WidgetFactory,
                 signal_inspector: Optional[SignalInspector] = None):
        
        self.config = config
        self.theme_manager = theme_manager
        self.shortcut_manager = shortcut_manager
        self.orchestrator = orchestrator
        self.widget_factory = widget_factory
        self.signal_inspector = signal_inspector
        self.signals = UISignals()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app = QApplication.instance()
        self.main_window: Optional[MainWindow] = None
        self.settings_window: Optional[SettingsWindow] = None
        self.about_window: Optional[AboutWindow] = None
        
        self.recent_documents: List[str] = []
        self.open_documents: Dict[str, Any] = {}
        
        self._create_main_window()
        self._connect_signals()
        self._load_recent_documents()
    
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
        self.signals.follow_up_requested.connect(self.on_follow_up)
        
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
        
        # Add to recent documents
        if path not in self.recent_documents:
            self.recent_documents.insert(0, path)
            self.recent_documents = self.recent_documents[:self.config.documents.max_recent_files]
            self._save_recent_documents()
    
    def _on_document_closed(self, doc_id: str):
        """Handle document closed"""
        self.logger.info(f"Document closed with ID: {doc_id}")
        if doc_id in self.open_documents:
            del self.open_documents[doc_id]
    
    def _on_voice_input(self, text: str):
        """Handle voice input"""
        self.logger.info(f"Voice input received: {text}")
        # Process voice input through orchestrator
        # This would trigger the appropriate action
    
    def _load_recent_documents(self):
        """Load recent documents from file"""
        import json
        from pathlib import Path
        
        recent_file = Path.home() / '.doc_explainer' / 'recent.json'
        if recent_file.exists():
            try:
                with open(recent_file, 'r') as f:
                    data = json.load(f)
                    self.recent_documents = data.get('documents', [])
            except Exception as e:
                self.logger.error(f"Error loading recent documents: {e}")
        else:
            self.logger.info("No recent documents found.")
    
    def _save_recent_documents(self):
        """Save recent documents to file"""
        import json
        from pathlib import Path
        
        recent_file = Path.home() / '.doc_explainer' / 'recent.json'
        try:
            with open(recent_file, 'w') as f:
                json.dump({'documents': self.recent_documents}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving recent documents: {e}")
    
    def get_recent_documents(self) -> List[str]:
        """Get list of recent documents"""
        return self.recent_documents
    
    def on_document_registered(self, path: str) -> str:
        """Handle document registration"""
        self.logger.info(f"Registering document: {path}")
        response = self.orchestrator.register_document(path)

        if response.success and type(response) == DocumentResponse and response.doc_id:
            self.logger.info(f"Document registered with ID: {response.doc_id}")
            return response.doc_id
    
        return "I couldn't register the document. Please check the file and try again."
    
    def on_explain(self, doc_id: int, text: str, page: int, position: int):
        """Handle explain action"""
        self.logger.info(f"Explain requested for doc {doc_id}")
        section_id = self.orchestrator.get_section_id_at_position(
            str(doc_id), page, position
        )
        response = self.orchestrator.explain(
            doc_id=str(doc_id),
            selected_text=text,
            section_id=section_id
        )
        if response.success and type(response) == ExplainResponse and response.explanation:
            self.main_window.sidebar.update_explanation(
                response.explanation, section_id
            )
        else:
            self.logger.warning(f"Explain failed for doc {doc_id} at section {section_id}")
    
    def on_summarize(self, doc_id: int, text: str, page: int, position: int):
        """Handle summarize action"""
        self.logger.info(f"Summarize requested for doc {doc_id}")
        section_id = self.orchestrator.get_section_id_at_position(
            str(doc_id), page, position
        )
        response = self.orchestrator.summarize(
            doc_id=str(doc_id),
            selected_text=text,
            section_id=section_id
        )
        if response.success and  type(response) == SummarizeResponse and response.explanation:
            self.main_window.sidebar.update_explanation(
                response.explanation, section_id
            )
        else:
            self.logger.warning(f"Summarize failed for doc {doc_id} at section {section_id}")
    
    def on_ask(self, doc_id: int, text: str, page: int, position: int):
        """Handle ask action"""
        self.logger.info(f"Ask requested for doc {doc_id}")
        section_id = self.orchestrator.get_section_id_at_position(
            str(doc_id), page, position
        )
        response = self.orchestrator.answer(
            doc_id=str(doc_id),
            question=text,
            section_id=section_id
        )
        if response.success and type(response) == AnswerResponse and response.explanation:
            self.main_window.sidebar.update_explanation(
                response.explanation, section_id
            )
        else:
            self.logger.warning(f"Ask failed for doc {doc_id} at section {section_id}")
    
    def on_follow_up(self, doc_id: int, question: str, section_id: int):
        """Handle follow-up question"""
        self.logger.info(f"Follow-up question for doc {doc_id}: {question}")
        response = self.orchestrator.answer(
            doc_id=str(doc_id),
            question=question,
            section_id=section_id
        )
        if response.success and  type(response) == AnswerResponse and response.explanation:
            self.main_window.sidebar.update_explanation(
                response.explanation, section_id
            )
        else:
            self.logger.warning(f"Follow-up question failed for doc {doc_id} at section {section_id}")
    
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
        # Save configuration
        self.config.save()
        
        # Close all windows
        if self.main_window:
            self.main_window.close()
        
        if self.settings_window:
            self.settings_window.close()
        
        if self.about_window:
            self.about_window.close()
        
        # Quit application
        self.app.quit()


