#!/usr/bin/env python3
"""
Doc Explainer Application
Main entry point for the GUI application
"""

import sys
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon

from ..managers.window_manager import WindowManager
from ..managers.theme_manager import ThemeManager
from ..managers.shortcut_manager import ShortcutManager
from ..config import UIConfig
from ..styles.theme import LightTheme, DarkTheme, HighContrastTheme, SepiaTheme
from ..utils.file_utils import FileUtils
from ..utils.signal_utils import SignalInspector
from ..factories.widget_factory import WidgetFactory
from src.orchestrator import DocExplainerOrchestrator, OrchestratorConfig


class DocExplainerApp:
    """Main application class"""
    app: QApplication
    window_manager: WindowManager
    theme_manager: ThemeManager 
    shortcut_manager: ShortcutManager
    widget_factory: WidgetFactory 
    signal_inspector: SignalInspector
    orchestrator: DocExplainerOrchestrator
    config: UIConfig
    logger: logging.Logger

    def __init__(self, config_path: str = '', debug: bool = False):
        # Import orchestrator
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Ensure user directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            Path.home() / '.doc_explainer',
            Path.home() / '.doc_explainer' / 'cache',
            Path.home() / '.doc_explainer' / 'logs',
            Path.home() / '.doc_explainer' / 'config',
            Path.home() / '.doc_explainer' / 'themes',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str = None) -> UIConfig:
        """Load configuration"""
        self.logger.info("Loading configuration")
        config = UIConfig()
        
        # Try loading from default location first
        default_config = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        if default_config.exists():
            try:
                with open(default_config, 'r') as f:
                    config_dict = json.load(f)
                    config = UIConfig.from_dict(config_dict)
                self.logger.info(f"Loaded config from {default_config}")
            except Exception as e:
                self.logger.error(f"Error loading config from {default_config}: {e}")
        
        # Override with provided config file
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    config = UIConfig.from_dict(config_dict)
                self.logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {e}")
        
        return config
    
    def _setup_application(self):
        """Setup Qt application"""
        # Set application attributes
        QApplication.setApplicationName("Doc Explainer")
        QApplication.setOrganizationName("DocExplainer")
        QApplication.setApplicationVersion("1.0.0")
        
        # Set application icon (if available)
        icon_path = Path.home() / '.doc_explainer' / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            self.logger.info(f"Setting application icon from {icon_path}")
            self.app.setWindowIcon(QIcon(str(icon_path)))
        else:
            self.logger.warning(f"App icon not found at {icon_path}, using default icon")
    
    def _show_splash_screen(self) -> Optional[QSplashScreen]:
        """Show splash screen on startup"""
        splash_path =  Path.home() / '.doc_explainer' / "resources" / "images" / "splash.png"
        self.logger.info("Showing splash screen")
        self.logger.debug(f"Looking for splash image at {splash_path}")
        if splash_path.exists():
            splash_pix = QPixmap(str(splash_path))
            splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
            splash.show()
            self.app.processEvents()
            return splash
        self.logger.warning("Splash image not found, skipping splash screen")
        return None
    
    def _init_components(self, splash: Optional[QSplashScreen] = None):
        """Initialize all components"""
        components = [
            ("Initializing UI...", self._init_ui_components),
            ("Loading themes...", self._init_themes),
            ("Setting up shortcuts...", self._init_shortcuts),
            ("Initializing orchestrator...", self._init_orchestrator),
            ("Creating window manager...", self._init_window_manager),
            ("Setting up factories...", self._init_factories),
        ]
        
        for message, init_func in components:
            if splash:
                splash.showMessage(message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
                self.app.processEvents()
            self.logger.info(message)
            init_func()
    
    def _init_ui_components(self):
        """Initialize UI components"""
        self.signal_inspector = SignalInspector(enabled=self.debug)
        self.widget_factory = WidgetFactory()
    
    def _init_themes(self):
        """Initialize theme manager"""
        self.theme_manager = ThemeManager(self.app)
        
        # Register built-in themes
        self.theme_manager.add_theme("light", LightTheme())
        self.theme_manager.add_theme("dark", DarkTheme())
        self.theme_manager.add_theme("high_contrast", HighContrastTheme())
        self.theme_manager.add_theme("sepia", SepiaTheme())
        
        # Load custom themes from directory
        themes_dir = Path.home() / '.doc_explainer' / 'themes'
        if themes_dir.exists():
            for theme_file in themes_dir.glob("*.json"):
                try:
                    with open(theme_file, 'r') as f:
                        theme_data = json.load(f)
                        # This would need proper theme class creation
                        self.logger.info(f"Loaded custom theme from {theme_file}")
                except Exception as e:
                    self.logger.error(f"Error loading theme {theme_file}: {e}")
        
        # Apply configured theme
        self.theme_manager.set_theme(self.config.theme)
    
    def _init_shortcuts(self):
        """Initialize shortcut manager"""
        self.shortcut_manager = ShortcutManager()

    def _init_viewers(self):
        """Initialize viewer factories"""
        self.view_factory = ViewerFactory()

    def _init_orchestrator(self):
        """Initialize orchestrator"""
        orchestrator_config = OrchestratorConfig(
            llm_provider=self.config.llm_provider,
            temperature=self.config.llm_temperature,
            persist_directory=str(Path.home() / '.doc_explainer' / 'cache'),
            enable_knowledge_graph=self.config.kg_enabled,
            enable_memory=self.config.memory_enabled,
            enable_session_tracking=self.config.session_tracking,
            llm_kwargs={
                "max_tokens": self.config.llm_max_tokens,
                "timeout": self.config.llm_timeout,
            }
        )
        
        self.orchestrator = DocExplainerOrchestrator(config=orchestrator_config)
    
    def _init_window_manager(self):
        """Initialize window manager"""
        self.window_manager = WindowManager(
            config=self.config,
            theme_manager=self.theme_manager,
            shortcut_manager=self.shortcut_manager,
            orchestrator=self.orchestrator,
            widget_factory=self.widget_factory,
            signal_inspector=self.signal_inspector
        )
    
    def _init_factories(self):
        """Initialize factories"""
        # Register custom viewers if needed
        # ViewerFactory.register_viewer('.custom', CustomViewer)
        pass
    
    def _check_for_updates(self):
        """Check for updates (async)"""
        # This would connect to a update server
        pass
    
    def _load_recent_documents(self):
        """Load recently opened documents"""
        recent_file = Path.home() / '.doc_explainer' / 'recent.json'
        if recent_file.exists() and self.config.open_last_docs:
            try:
                with open(recent_file, 'r') as f:
                    recent_docs = json.load(f)
                
                for doc_path in recent_docs.get('documents', [])[:5]:
                    if Path(doc_path).exists():
                        self.logger.info(f"Loading recent document: {doc_path}")
                        self.window_manager.on_document_registered(doc_path)
            except Exception as e:
                self.logger.error(f"Error loading recent documents: {e}")

    def _setup_exception_handling(self):
        """Setup global exception handling"""
        def excepthook(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions"""
            self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText("An unexpected error occurred")
            error_dialog.setDetailedText(str(exc_value))
            error_dialog.exec()
        
        sys.excepthook = excepthook
    
    def run(self, document_path: str = None) -> int:
        """Run the application"""
        try:
            # Create Qt application
            self.app = QApplication(sys.argv)
            self._setup_application()
            self._setup_exception_handling()
            
            # Show splash screen
            splash = self._show_splash_screen()
            
            # Initialize components
            self._init_components(splash)
            
            # Launch main window
            self.window_manager.launch()

            if document_path:
                self.logger.info(f"Opening document from command line: {document_path}")
                self.window_manager.on_document_registered(document_path)
            
            # Close splash screen
            if splash:
                splash.finish(self.window_manager.main_window)
            
            # Load recent documents if enabled
            if self.config.open_last_docs:
                QTimer.singleShot(100, self._load_recent_documents)
            
            # Check for updates
            if self.config.check_updates:
                QTimer.singleShot(2000, self._check_for_updates)
            
            self.logger.info("Application started successfully")
            
            # Run application
            return self.app.exec()
            
        except Exception as e:
            self.logger.exception("Fatal error during application startup")
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Startup Error")
            error_dialog.setText("Failed to start application")
            error_dialog.setDetailedText(str(e))
            error_dialog.exec()
            
            return 1
    
    def cleanup(self):
        """Cleanup application resources"""
        self.logger.info("Cleaning up application resources")
        
        # Save recent documents
        if self.window_manager:
            recent_docs = self.window_manager.get_recent_documents()
            recent_file = Path.home() / '.doc_explainer' / 'recent.json'
            try:
                with open(recent_file, 'w') as f:
                    json.dump({'documents': recent_docs}, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error saving recent documents: {e}")
        
        # Save configuration
        config_file = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")