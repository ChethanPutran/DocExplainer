from typing import Dict, Optional
from .theme import Theme


class StyleSheet:
    """Generates QSS style sheets from themes"""
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
    
    def generate(self, theme: Theme) -> str:
        """Generate style sheet from theme"""
        cache_key = id(theme)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        css = f"""
        /* Main Window */
        QMainWindow {{
            background-color: {theme.background_primary};
        }}
        
        /* Dock Widgets */
        QDockWidget {{
            font-size: 12pt;
            titlebar-close-icon: url(none);
            titlebar-normal-icon: url(none);
        }}
        
        QDockWidget::title {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
            padding: 5px;
            text-align: left;
        }}
        
        /* Tab Widget */
        QTabWidget::pane {{
            border: 1px solid {theme.border_color};
            background-color: {theme.background_primary};
        }}
        
        QTabBar::tab {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {theme.background_tertiary};
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {theme.accent_secondary};
        }}
        
        QPushButton:pressed {{
            background-color: {theme.accent_tertiary};
        }}
        
        QPushButton:disabled {{
            background-color: {theme.background_tertiary};
            color: {theme.text_disabled};
        }}
        
        /* Toolbar */
        QToolBar {{
            background-color: {theme.background_secondary};
            border: none;
            spacing: 3px;
            padding: 3px;
        }}
        
        QToolBar QToolButton {{
            background-color: transparent;
            color: {theme.text_primary};
            border-radius: 3px;
            padding: 5px;
        }}
        
        QToolBar QToolButton:hover {{
            background-color: {theme.background_tertiary};
        }}
        
        QToolBar QToolButton:pressed {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
        }}
        
        /* Scroll Areas */
        QScrollArea {{
            border: none;
            background-color: {theme.background_primary};
        }}
        
        QScrollBar:vertical {{
            background-color: {theme.background_secondary};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {theme.accent_primary};
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {theme.accent_secondary};
        }}
        
        QScrollBar:horizontal {{
            background-color: {theme.background_secondary};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {theme.accent_primary};
            min-width: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {theme.accent_secondary};
        }}
        
        /* Labels */
        QLabel {{
            color: {theme.text_primary};
        }}
        
        /* Line Edits */
        QLineEdit {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            padding: 5px;
        }}
        
        QLineEdit:focus {{
            border: 2px solid {theme.accent_primary};
        }}
        
        /* Text Edit */
        QTextEdit {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            border-radius: 4px;
        }}
        
        /* List Widgets */
        QListWidget {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            outline: none;
        }}
        
        QListWidget::item {{
            padding: 5px;
            border-bottom: 1px solid {theme.border_color};
        }}
        
        QListWidget::item:selected {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
        }}
        
        QListWidget::item:hover:!selected {{
            background-color: {theme.background_tertiary};
        }}
        
        /* Progress Bar */
        QProgressBar {{
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            text-align: center;
            background-color: {theme.background_secondary};
        }}
        
        QProgressBar::chunk {{
            background-color: {theme.accent_primary};
            border-radius: 3px;
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {theme.background_secondary};
            color: {theme.text_secondary};
            border-top: 1px solid {theme.border_color};
        }}
        
        QStatusBar QLabel {{
            color: {theme.text_secondary};
        }}
        
        /* Menu Bar */
        QMenuBar {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border-bottom: 1px solid {theme.border_color};
        }}
        
        QMenuBar::item:selected {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
        }}
        
        QMenu {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
        }}
        
        QMenu::item:selected {{
            background-color: {theme.accent_primary};
            color: {theme.text_on_accent};
        }}
        
        /* Group Box */
        QGroupBox {{
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            color: {theme.text_primary};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        
        /* Check Box */
        QCheckBox {{
            color: {theme.text_primary};
        }}
        
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border: 1px solid {theme.border_color};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {theme.accent_primary};
            border-color: {theme.accent_primary};
        }}
        
        /* Radio Button */
        QRadioButton {{
            color: {theme.text_primary};
        }}
        
        QRadioButton::indicator {{
            width: 15px;
            height: 15px;
            border: 1px solid {theme.border_color};
            border-radius: 8px;
        }}
        
        QRadioButton::indicator:checked {{
            background-color: {theme.accent_primary};
            border-color: {theme.accent_primary};
        }}
        
        /* Spin Box */
        QSpinBox, QDoubleSpinBox {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            padding: 3px;
        }}
        
        /* Combo Box */
        QComboBox {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            border-radius: 4px;
            padding: 5px;
        }}
        
        QComboBox:hover {{
            border-color: {theme.accent_primary};
        }}
        
        QComboBox::drop-down {{
            border: none;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {theme.background_secondary};
            color: {theme.text_primary};
            border: 1px solid {theme.border_color};
            selection-background-color: {theme.accent_primary};
            selection-color: {theme.text_on_accent};
        }}
        """
        
        self.cache[cache_key] = css
        return css
    
    def clear_cache(self):
        """Clear style sheet cache"""
        self.cache.clear()