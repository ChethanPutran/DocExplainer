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
import yaml

from doc_explainer.config.logger import setup_logging

from ..managers.window_manager import WindowManager
from ..managers.theme_manager import ThemeManager
from ..managers.shortcut_manager import ShortcutManager
from ....config import UIConfig, LLMConfig, BackendConfig
from ..styles.theme import LightTheme, DarkTheme, HighContrastTheme, SepiaTheme
from ..utils.file_utils import FileUtils
from ..utils.signal_utils import SignalInspector
from ..factories.widget_factory import WidgetFactory
from ....orchestrator import DocExplainerOrchestrator, OrchestratorConfig
from ..factories.viewer_factory import ViewerFactory
import logging 

logger = logging.getLogger(__name__)

class DocExplainerApp:
    """Main application class"""
    app: Optional[QApplication] = None
    window_manager: Optional[WindowManager] = None
    theme_manager: Optional[ThemeManager] = None
    view_factory: Optional[ViewerFactory] = None
    shortcut_manager: Optional[ShortcutManager] = None
    widget_factory: Optional[WidgetFactory] = None
    signal_inspector: Optional[SignalInspector] = None
    orchestrator: Optional[DocExplainerOrchestrator] = None
    ui_config: Optional[UIConfig] = None
    llm_config: Optional[LLMConfig] = None
    backend_config: Optional[BackendConfig] = None
    logger: Optional[logging.Logger] = None

    def __init__(self, 
                 config: Optional[Path] = None, 
                 log_level: str = 'INFO',
                 reset_config: bool = False,
                 clear_cache: bool = False):
        # Import orchestrator
        self.debug = log_level.upper() == 'DEBUG'
        
        self.config = config
        self.ui_config, self.llm_config, self.backend_config = self._load_config(config)
        
        # Ensure user directories exist
        self._ensure_directories()

        # Setup logging
        setup_logging(self.backend_config.logging)

        if(reset_config):
            self.handle_reset_config()
            sys.exit(0)

        if(clear_cache):
            self.handle_clear_cache()
            sys.exit(0)

    def handle_reset_config(self):
        """Reset configuration to defaults"""
        config_file = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        if config_file.exists():
            config_file.unlink()
            logger.info("Configuration reset to defaults")
        else:
            logger.info("No configuration file found")


    def handle_clear_cache(self):
        """Clear application cache"""
        cache_dir = Path.home() / '.doc_explainer' / 'cache'
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            logger.info("Cache cleared")
        else:
            logger.info("No cache directory found")
    
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
    
    def _load_config(
        self,
        config: Optional[Path] = None
    ) -> tuple[UIConfig, LLMConfig, BackendConfig]:
        """
        Load configuration with the following priority:

        1. Explicitly provided config path
        2. User config: ~/.doc_explainer/config/
        3. Project config: <project_root>/config/
        4. Config class defaults if no file exists
        """

        logger.info("Loading configuration")

        default_config_dir = Path.home() / ".doc_explainer" / "config"

        # Project root:
        # src/doc_explainer/ui/gui/app/app.py
        # parents[0] = app
        # parents[1] = gui
        # parents[2] = ui
        # parents[3] = doc_explainer
        # parents[4] = src
        # parents[5] = project root
        project_root = Path(__file__).resolve().parents[5]
        project_config_dir = project_root / "config"

        logger.debug(
            "Project config directory: %s",
            project_config_dir
        )

        # ---------------------------------------------------------
        # Resolve individual config paths
        # ---------------------------------------------------------

        def resolve_config_path(
            name: str,
            explicit_path: Optional[Path] = None
        ) -> Path:
            """
            Resolve config path using:

            explicit path
                ↓
            ~/.doc_explainer/config
                ↓
            project/config
            """

            # 1. Explicitly provided path
            if explicit_path is not None:
                path = Path(explicit_path).expanduser()

                if path.exists():
                    logger.info(
                        "Using explicitly provided %s config: %s",
                        name,
                        path
                    )
                    return path

                logger.warning(
                    "Explicit %s config does not exist: %s",
                    name,
                    path
                )

                # Important:
                # If the user explicitly provided a path, don't silently
                # switch to another configuration.
                return path

            # 2. User configuration
            user_path = default_config_dir / f"{name}_config.yaml"

            if user_path.exists():
                logger.info(
                    "Using user %s config: %s",
                    name,
                    user_path
                )
                return user_path

            # 3. Project configuration
            project_path = project_config_dir / f"{name}_config.yaml"

            if project_path.exists():
                logger.info(
                    "Using project default %s config: %s",
                    name,
                    project_path
                )
                return project_path

            # Nothing exists.
            # Return the user path because the config class can create
            # the default config there if desired.
            logger.info(
                "No %s config found. Using user config path: %s",
                name,
                user_path
            )

            return user_path

        # ---------------------------------------------------------
        # Extract explicit paths
        # ---------------------------------------------------------

        explicit_ui = None
        explicit_llm = None
        explicit_backend = None

        if config is not None:
            explicit_ui = config / "ui_config.yaml"
            explicit_llm = config / "llm_config.yaml"
            explicit_backend = config / "backend_config.yaml"

        # ---------------------------------------------------------
        # Resolve paths
        # ---------------------------------------------------------

        ui_config_path = resolve_config_path(
            "ui",
            explicit_ui
        )

        llm_config_path = resolve_config_path(
            "llm",
            explicit_llm
        )

        backend_config_path = resolve_config_path(
            "backend",
            explicit_backend
        )

        logger.info("UI config path: %s", ui_config_path)
        logger.info("LLM config path: %s", llm_config_path)
        logger.info("Backend config path: %s", backend_config_path)

        # ---------------------------------------------------------
        # Load configs
        # ---------------------------------------------------------

        ui_config = UIConfig.load(
            filepath=str(ui_config_path)
        )

        llm_config = LLMConfig.load(
            filepath=str(llm_config_path)
        )

        backend_config = BackendConfig.load(
            filepath=str(backend_config_path)
        )

        return ui_config, llm_config, backend_config
    
    def _setup_application(self):
        """Setup Qt application"""
        # Set application attributes
        QApplication.setApplicationName("Doc Explainer")
        QApplication.setOrganizationName("DocExplainer")
        QApplication.setApplicationVersion("1.0.0")
        
        # Set application icon (if available)
        icon_path = Path.home() / '.doc_explainer' / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            logger.info(f"Setting application icon from {icon_path}")
            self.app.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(f"App icon not found at {icon_path}, using default icon")
    
    def _show_splash_screen(self) -> Optional[QSplashScreen]:
        """Show splash screen on startup"""
        splash_path =  Path.home() / '.doc_explainer' / "resources" / "images" / "splash.png"
        logger.info("Showing splash screen")
        logger.debug(f"Looking for splash image at {splash_path}")
        if splash_path.exists():
            splash_pix = QPixmap(str(splash_path))
            splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
            splash.show()
            self.app.processEvents()
            return splash
        logger.warning("Splash image not found, skipping splash screen")
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
            logger.info(message)
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
                        logger.info(f"Loaded custom theme from {theme_file}")
                except Exception as e:
                    logger.error(f"Error loading theme {theme_file}: {e}")
        
        # Apply configured theme
        self.theme_manager.set_theme(self.ui_config.theme.name)
    
    def _init_shortcuts(self):
        """Initialize shortcut manager"""
        self.shortcut_manager = ShortcutManager()

    def _init_viewers(self):
        """Initialize viewer factories"""
        self.view_factory = ViewerFactory()

    def _init_orchestrator(self):
        """Initialize orchestrator"""
        orchestrator_config = OrchestratorConfig(
            llm=self.llm_config,
            backend=self.backend_config,
        )
        
        self.orchestrator = DocExplainerOrchestrator(config=orchestrator_config)

    def _init_window_manager(self):
        """Initialize window manager"""
        self.window_manager = WindowManager(
            config=self.ui_config,
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
        if recent_file.exists() and self.ui_config.startup.open_last_docs:
            try:
                with open(recent_file, 'r') as f:
                    recent_docs = json.load(f)
                
                for doc_path in recent_docs.get('documents', [])[:5]:
                    if Path(doc_path).exists():
                        logger.info(f"Loading recent document: {doc_path}")
                        self.window_manager.on_document_registered(doc_path)
            except Exception as e:
                logger.error(f"Error loading recent documents: {e}")

    def _setup_exception_handling(self):
        """Setup global exception handling"""
        def excepthook(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions"""
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText("An unexpected error occurred")
            error_dialog.setDetailedText(str(exc_value))
            error_dialog.exec()
        
        sys.excepthook = excepthook
    
    def run(self, document_path: Optional[str] = None) -> int:
        """Run the application"""
        try:
            # Create Qt application
            self.app = QApplication(sys.argv)
            self._setup_application()
            self._setup_exception_handling()
            
            # Show splash screen
            if not self.ui_config.system.show_splash:
                splash = None
            else:
                splash = self._show_splash_screen()
            
            # Initialize components
            self._init_components(splash)
            
            # Launch main window
            self.window_manager.launch()

            if document_path:
                logger.info(f"Opening document from command line: {document_path}")
                self.window_manager.on_document_registered(document_path)
            
            # Close splash screen
            if splash and self.window_manager.main_window is not None:
                splash.finish(self.window_manager.main_window)
            
            # Load recent documents if enabled
            if self.ui_config.startup.open_last_docs:
                QTimer.singleShot(100, self._load_recent_documents)
            
            # Check for updates
            if self.ui_config.startup.check_updates:
                QTimer.singleShot(2000, self._check_for_updates)
            
            logger.info("Application started successfully")
            
            # Run application
            return self.app.exec()
            
        except Exception as e:
            logger.exception("Fatal error during application startup")
            
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
        logger.info("Cleaning up application resources")
        
        # Save recent documents
        if self.window_manager:
            self.window_manager.shutdown()
            recent_docs = self.window_manager.get_recent_documents()
            recent_file = Path.home() / '.doc_explainer' / 'recent.json'
            try:
                with open(recent_file, 'w') as f:
                    json.dump({'documents': recent_docs}, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving recent documents: {e}")
        
        # Save configuration
        self.ui_config.save()
        self.llm_config.save()
        self.backend_config.save()
