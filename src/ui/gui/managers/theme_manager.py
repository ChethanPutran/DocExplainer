from typing import Dict, Optional, Callable
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import QObject, Signal

from ..styles.theme import Theme, LightTheme, DarkTheme, HighContrastTheme
from ..styles.style_sheet import StyleSheet


class ThemeManager(QObject):
    """Manages application themes"""
    
    theme_changed = Signal(str)  # theme_name
    
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.themes: Dict[str, Theme] = {
            'light': LightTheme(),
            'dark': DarkTheme(),
            'high_contrast': HighContrastTheme()
        }
        self.current_theme: str = 'light'
        self.style_sheet = StyleSheet()
        self.theme_observers: Dict[str, list] = {}
    
    def set_theme(self, theme_name: str) -> bool:
        """Set active theme"""
        if theme_name not in self.themes:
            return False
        
        self.current_theme = theme_name
        theme = self.themes[theme_name]
        
        # Apply theme
        self._apply_palette(theme)
        self._apply_style_sheet(theme)
        
        # Notify observers
        self.theme_changed.emit(theme_name)
        self._notify_observers(theme_name)
        
        return True
    
    def _apply_palette(self, theme: Theme):
        """Apply color palette"""
        palette = QPalette()
        
        # Set colors based on theme
        palette.setColor(QPalette.Window, QColor(theme.background_primary))
        palette.setColor(QPalette.WindowText, QColor(theme.text_primary))
        palette.setColor(QPalette.Base, QColor(theme.background_secondary))
        palette.setColor(QPalette.AlternateBase, QColor(theme.background_tertiary))
        palette.setColor(QPalette.ToolTipBase, QColor(theme.background_primary))
        palette.setColor(QPalette.ToolTipText, QColor(theme.text_primary))
        palette.setColor(QPalette.Text, QColor(theme.text_primary))
        palette.setColor(QPalette.Button, QColor(theme.accent_primary))
        palette.setColor(QPalette.ButtonText, QColor(theme.text_on_accent))
        palette.setColor(QPalette.BrightText, QColor(theme.accent_secondary))
        palette.setColor(QPalette.Link, QColor(theme.accent_primary))
        palette.setColor(QPalette.Highlight, QColor(theme.accent_primary))
        palette.setColor(QPalette.HighlightedText, QColor(theme.text_on_accent))
        
        self.app.setPalette(palette)
    
    def _apply_style_sheet(self, theme: Theme):
        """Apply style sheet"""
        css = self.style_sheet.generate(theme)
        self.app.setStyleSheet(css)
    
    def register_observer(self, widget_id: str, callback: Callable[[str], None]):
        """Register observer for theme changes"""
        if widget_id not in self.theme_observers:
            self.theme_observers[widget_id] = []
        self.theme_observers[widget_id].append(callback)
    
    def _notify_observers(self, theme_name: str):
        """Notify all observers of theme change"""
        for callbacks in self.theme_observers.values():
            for callback in callbacks:
                try:
                    callback(theme_name)
                except Exception as e:
                    print(f"Error notifying observer: {e}")
    
    def get_current_theme(self) -> Theme:
        """Get current theme object"""
        return self.themes[self.current_theme]
    
    def get_theme_names(self) -> list:
        """Get list of available theme names"""
        return list(self.themes.keys())
    
    def add_theme(self, name: str, theme: Theme):
        """Add a new theme"""
        self.themes[name] = theme
    
    def remove_theme(self, name: str) -> bool:
        """Remove a theme"""
        if name in self.themes and name != self.current_theme:
            del self.themes[name]
            return True
        return False