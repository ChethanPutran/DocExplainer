from typing import Dict, Optional, Callable, List, Tuple
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QKeySequence, QShortcut, QAction
from PySide6.QtCore import QObject


class ShortcutManager(QObject):
    """Manages keyboard shortcuts"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.parent = parent
        self.shortcuts: Dict[str, QShortcut] = {}
        self.actions: Dict[str, QAction] = {}
        self.shortcut_map: Dict[str, Tuple[str, str]] = {}  # key -> (description, callback_id)
    
    def register_shortcut(self, key: str, description: str, 
                          callback: Callable, parent: Optional[QWidget] = None) -> QShortcut:
        """Register a keyboard shortcut"""
        parent = parent or self.parent
        
        if not parent:
            raise ValueError("Parent widget required for shortcut")
        
        key_sequence = QKeySequence(key)
        shortcut = QShortcut(key_sequence, parent)
        shortcut.activated.connect(callback)
        
        self.shortcuts[key] = shortcut
        self.shortcut_map[key] = (description, str(id(callback)))
        
        return shortcut
    
    def register_action(self, name: str, key: str, description: str,
                        callback: Callable, parent: Optional[QWidget] = None) -> QAction:
        """Register an action with shortcut"""
        action = QAction(description, parent)
        action.setShortcut(QKeySequence(key))
        action.triggered.connect(callback)
        
        self.actions[name] = action
        self.shortcut_map[key] = (description, name)
        
        return action
    
    def register_standard_shortcuts(self):
        """Register standard application shortcuts"""
        shortcuts = [
            ('Ctrl+O', 'Open Document', self._on_open),
            ('Ctrl+W', 'Close Tab', self._on_close_tab),
            ('Ctrl+S', 'Save', self._on_save),
            ('Ctrl+F', 'Find', self._on_find),
            ('Ctrl+Plus', 'Zoom In', self._on_zoom_in),
            ('Ctrl+Minus', 'Zoom Out', self._on_zoom_out),
            ('Ctrl+0', 'Reset Zoom', self._on_zoom_reset),
            ('Ctrl+B', 'Toggle Sidebar', self._on_toggle_sidebar),
            ('F5', 'Refresh', self._on_refresh),
            ('F1', 'Help', self._on_help),
            ('Ctrl+Q', 'Quit', self._on_quit),
        ]
        
        for key, desc, callback in shortcuts:
            self.register_shortcut(key, desc, callback)
    
    def get_shortcut_description(self, key: str) -> Optional[str]:
        """Get description for a shortcut"""
        if key in self.shortcut_map:
            return self.shortcut_map[key][0]
        return None
    
    def get_all_shortcuts(self) -> Dict[str, str]:
        """Get all registered shortcuts with descriptions"""
        return {key: self.shortcut_map[key][0] 
                for key in self.shortcut_map}
    
    def remove_shortcut(self, key: str) -> bool:
        """Remove a shortcut"""
        if key in self.shortcuts:
            self.shortcuts[key].setEnabled(False)
            self.shortcuts[key].deleteLater()
            del self.shortcuts[key]
            del self.shortcut_map[key]
            return True
        return False
    
    def enable_shortcut(self, key: str, enabled: bool = True):
        """Enable or disable a shortcut"""
        if key in self.shortcuts:
            self.shortcuts[key].setEnabled(enabled)
    
    def enable_all(self, enabled: bool = True):
        """Enable or disable all shortcuts"""
        for shortcut in self.shortcuts.values():
            shortcut.setEnabled(enabled)
    
    # Default handlers (to be connected by main window)
    def _on_open(self):
        pass
    
    def _on_close_tab(self):
        pass
    
    def _on_save(self):
        pass
    
    def _on_find(self):
        pass
    
    def _on_zoom_in(self):
        pass
    
    def _on_zoom_out(self):
        pass
    
    def _on_zoom_reset(self):
        pass
    
    def _on_toggle_sidebar(self):
        pass
    
    def _on_refresh(self):
        pass
    
    def _on_help(self):
        pass
    
    def _on_quit(self):
        pass