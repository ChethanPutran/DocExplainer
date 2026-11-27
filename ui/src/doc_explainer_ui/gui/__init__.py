from .app import DocExplainerApp
from .windows.main_window import MainWindow
from .managers.window_manager import WindowManager
from doc_explainer_ui.config import UIConfig
from .main import main
__all__ = [
    'DocExplainerApp',
    'MainWindow',
    'WindowManager',
    'UIConfig',
    'main'

]