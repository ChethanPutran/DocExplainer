#!/usr/bin/env python3
"""
Doc Explainer Application
Main entry point for the GUI application
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon

from .managers.window_manager import WindowManager
from .managers.theme_manager import ThemeManager
from .managers.shortcut_manager import ShortcutManager
from .config import UIConfig
from .styles.theme import LightTheme, DarkTheme, HighContrastTheme, SepiaTheme
from .utils.file_utils import FileUtils
from .utils.signal_utils import SignalInspector
from .factories.widget_factory import WidgetFactory
from .factories.viewer_factory import ViewerFactory
from .windows.about_window import AboutWindow
from .windows.settings_window import SettingsWindow
from .widgets.common.status_bar import StatusBar
from .widgets.common.toolbar import MainToolbar
from .widgets.sidebar.sidebar import Sidebar
from .widgets.voice.voice_input import VoiceInput
from .widgets.voice.voice_output import VoiceOutput
from .widgets.viewers.pdf_viewer import PDFViewer
from .widgets.viewers.text_viewer import TextViewer
from .widgets.viewers.html_viewer import HTMLViewer

# Import orchestrator
from src.orchestrator.orchestrator import DocExplainerOrchestrator
from src.orchestrator.config import OrchestratorConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / '.doc_explainer' / 'app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DocExplainerApp:
    """Main application class"""
    
    def __init__(self, config_path: str = None, debug: bool = False):
        self.debug = debug
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.app: Optional[QApplication] = None
        self.window_manager: Optional[WindowManager] = None
        self.theme_manager: Optional[ThemeManager] = None
        self.shortcut_manager: Optional[ShortcutManager] = None
        self.widget_factory: Optional[WidgetFactory] = None
        self.signal_inspector: Optional[SignalInspector] = None
        self.orchestrator: Optional[DocExplainerOrchestrator] = None
        
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
        config = UIConfig()
        
        # Try loading from default location first
        default_config = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        if default_config.exists():
            try:
                with open(default_config, 'r') as f:
                    config_dict = json.load(f)
                    config = UIConfig.from_dict(config_dict)
                logger.info(f"Loaded config from {default_config}")
            except Exception as e:
                logger.error(f"Error loading config from {default_config}: {e}")
        
        # Override with provided config file
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    config = UIConfig.from_dict(config_dict)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        return config
    
    def _setup_application(self):
        """Setup Qt application"""
        # Set application attributes
        QApplication.setApplicationName("Doc Explainer")
        QApplication.setOrganizationName("DocExplainer")
        QApplication.setApplicationVersion("1.0.0")
        
        # Set application icon (if available)
        icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            self.app.setWindowIcon(QIcon(str(icon_path)))
    
    def _show_splash_screen(self) -> Optional[QSplashScreen]:
        """Show splash screen on startup"""
        splash_path = Path(__file__).parent / "resources" / "images" / "splash.png"
        if splash_path.exists():
            splash_pix = QPixmap(str(splash_path))
            splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
            splash.show()
            self.app.processEvents()
            return splash
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
                splash.showMessage(message, Qt.AlignBottom | Qt.AlignCenter, Qt.white)
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
        self.theme_manager.set_theme(self.config.theme)
    
    def _init_shortcuts(self):
        """Initialize shortcut manager"""
        self.shortcut_manager = ShortcutManager()
    
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
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText("An unexpected error occurred")
            error_dialog.setDetailedInfo(str(exc_value))
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
            
            # Close splash screen
            if splash:
                splash.finish(self.window_manager.main_window)
            
            # Load recent documents if enabled
            if self.config.open_last_docs:
                QTimer.singleShot(100, self._load_recent_documents)
            
            # Check for updates
            if self.config.check_updates:
                QTimer.singleShot(2000, self._check_for_updates)
            
            logger.info("Application started successfully")
            
            # Run application
            return self.app.exec()
            
        except Exception as e:
            logger.exception("Fatal error during application startup")
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Startup Error")
            error_dialog.setText("Failed to start application")
            error_dialog.setDetailedInfo(str(e))
            error_dialog.exec()
            
            return 1
    
    def cleanup(self):
        """Cleanup application resources"""
        logger.info("Cleaning up application resources")
        
        # Save recent documents
        if self.window_manager:
            recent_docs = self.window_manager.get_recent_documents()
            recent_file = Path.home() / '.doc_explainer' / 'recent.json'
            try:
                with open(recent_file, 'w') as f:
                    json.dump({'documents': recent_docs}, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving recent documents: {e}")
        
        # Save configuration
        config_file = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Doc Explainer - Intelligent Document Explanation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Open PDF document
  %(prog)s --config custom.json document.pdf  # Use custom config
  %(prog)s --debug --no-splash document.html  # Debug mode without splash
  %(prog)s --theme dark document.txt           # Open with dark theme
        """
    )
    
    parser.add_argument(
        'document',
        nargs='?',
        help='Document to open (PDF, TXT, HTML, etc.)'
    )
    
    parser.add_argument(
        '--config',
        '-c',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--theme',
        '-t',
        choices=['light', 'dark', 'high_contrast', 'sepia'],
        help='Theme to use'
    )
    
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-splash',
        action='store_true',
        help='Disable splash screen'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--version',
        '-v',
        action='version',
        version='Doc Explainer 1.0.0'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--reset-config',
        action='store_true',
        help='Reset configuration to defaults'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear application cache'
    )
    
    return parser


def handle_reset_config():
    """Reset configuration to defaults"""
    config_file = Path.home() / '.doc_explainer' / 'config' / 'ui_config.json'
    if config_file.exists():
        config_file.unlink()
        print("Configuration reset to defaults")
    else:
        print("No configuration file found")


def handle_clear_cache():
    """Clear application cache"""
    cache_dir = Path.home() / '.doc_explainer' / 'cache'
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        print("Cache cleared")
    else:
        print("No cache directory found")


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.reset_config:
        handle_reset_config()
        return 0
    
    if args.clear_cache:
        handle_clear_cache()
        return 0
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Override theme from command line
    if args.theme:
        # This would override the config theme
        pass
    
    # Create and run application
    app = DocExplainerApp(
        config_path=args.config,
        debug=args.debug
    )
    
    # Handle profiling
    if args.profile:
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        exit_code = app.run(args.document)
        
        profiler.disable()
        
        # Save profile stats
        profiler.dump_stats(Path.home() / '.doc_explainer' / 'profile.stats')
        
        # Print top 20 functions
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(20)
        print(s.getvalue())
        
    else:
        exit_code = app.run(args.document)
    
    # Cleanup
    app.cleanup()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())