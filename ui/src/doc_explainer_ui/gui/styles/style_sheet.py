from typing import Dict
from .theme import Theme


class StyleSheet:
    """Generates the application QSS from a theme: flat, rounded, one accent colour."""

    def __init__(self):
        self.cache: Dict[int, str] = {}

    def generate(self, theme: Theme) -> str:
        key = (id(theme), theme.name)
        if key in self.cache:
            return self.cache[key]
        t, r = theme, theme.radius
        css = f"""
        * {{ font-family: {t.font_family}; font-size: {t.font_size}pt; }}
        QWidget {{ color: {t.text_primary}; }}
        QMainWindow, QDialog {{ background-color: {t.background_primary}; }}
        QLabel {{ background: transparent; }}
        QToolTip {{ background-color: {t.background_secondary}; color: {t.text_primary};
                   border: 1px solid {t.border_color}; padding: 4px 8px; border-radius: 4px; }}

        /* Toolbar */
        QToolBar {{ background-color: {t.background_secondary}; border: none;
                   border-bottom: 1px solid {t.border_color}; padding: 5px 8px; spacing: 4px; }}
        QToolBar::separator {{ background: {t.border_color}; width: 1px; margin: 5px 6px; }}
        QToolBar QToolButton {{ background: transparent; color: {t.text_primary}; border: 1px solid transparent;
                               border-radius: {r}px; padding: 6px 11px; }}
        QToolBar QToolButton:hover {{ background-color: {t.hover}; }}
        QToolBar QToolButton:pressed {{ background-color: {t.background_tertiary}; }}
        QToolBar QToolButton:checked {{ background-color: {t.accent_soft}; color: {t.accent_primary}; }}
        QToolBar QToolButton:disabled {{ color: {t.text_disabled}; }}
        QToolBar QToolButton::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: center right;
                                               right: 4px; width: 8px; }}
        QToolButton[popupMode="2"] {{ padding-right: 18px; }}

        /* Document tabs */
        QTabWidget::pane {{ border: none; border-top: 1px solid {t.border_color}; background: {t.background_primary}; }}
        QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
        QTabBar::tab {{ background: transparent; color: {t.text_secondary}; padding: 9px 16px; margin-right: 2px;
                       border: none; border-bottom: 2px solid transparent; }}
        QTabBar::tab:hover:!selected {{ color: {t.text_primary}; background-color: {t.hover};
                                       border-top-left-radius: {r}px; border-top-right-radius: {r}px; }}
        QTabBar::tab:selected {{ color: {t.accent_primary}; border-bottom: 2px solid {t.accent_primary}; font-weight: 600; }}

        /* Dock (the tutor sidebar) */
        QDockWidget {{ font-weight: 600; titlebar-close-icon: none; }}
        QDockWidget::title {{ background-color: {t.background_secondary}; color: {t.text_primary};
                             padding: 9px 12px; text-align: left; border-bottom: 1px solid {t.border_color}; }}
        QMainWindow::separator {{ background: {t.border_color}; width: 1px; height: 1px; }}
        QSplitter::handle {{ background: {t.border_color}; }}
        QSplitter::handle:hover {{ background: {t.accent_primary}; }}

        /* Buttons: primary (filled) for every QPushButton; flat variant via property flat="true" */
        QPushButton {{ background-color: {t.accent_primary}; color: {t.text_on_accent}; border: 1px solid transparent;
                      padding: 7px 16px; border-radius: {r}px; font-weight: 600; }}
        QPushButton:hover {{ background-color: {t.accent_secondary}; }}
        QPushButton:pressed {{ background-color: {t.accent_tertiary}; }}
        QPushButton:focus {{ border: 1px solid {t.accent_soft}; }}
        QPushButton:disabled {{ background-color: {t.background_tertiary}; color: {t.text_disabled}; }}
        QPushButton:flat, QPushButton[flat="true"] {{ background: transparent; color: {t.text_primary};
                      border: 1px solid {t.border_color}; font-weight: 500; }}
        QPushButton:flat:hover, QPushButton[flat="true"]:hover {{ background-color: {t.hover}; }}
        QDialogButtonBox QPushButton, QMessageBox QPushButton {{ min-width: 72px; }}
        QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: {r}px; padding: 4px 8px; }}
        QToolButton:hover {{ background-color: {t.hover}; }}
        QToolButton:checked {{ background-color: {t.accent_soft}; color: {t.accent_primary}; }}
        QToolButton:disabled {{ color: {t.text_disabled}; }}

        /* Inputs */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit, QTextBrowser {{
            background-color: {t.background_secondary}; color: {t.text_primary}; border: 1px solid {t.border_color};
            border-radius: {r}px; padding: 6px 9px; selection-background-color: {t.accent_primary};
            selection-color: {t.text_on_accent}; }}
        QTextEdit, QPlainTextEdit, QTextBrowser {{ border-radius: 0; border: none; padding: 10px; }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {t.accent_primary}; }}
        QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {t.text_disabled}; background: {t.background_tertiary}; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{ background: {t.background_secondary}; border: 1px solid {t.border_color};
                                      selection-background-color: {t.accent_soft}; selection-color: {t.text_primary};
                                      outline: none; padding: 4px; }}
        QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            border: none; width: 16px; background: transparent; }}

        QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
        QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {t.text_secondary};
                                                        background: {t.background_secondary}; }}
        QCheckBox::indicator {{ border-radius: 4px; }}
        QRadioButton::indicator {{ border-radius: 9px; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {t.accent_primary}; border-color: {t.accent_primary}; }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent_primary}; }}

        QGroupBox {{ border: 1px solid {t.border_color}; border-radius: {r}px; margin-top: 12px; padding: 12px 8px 8px 8px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; color: {t.text_secondary}; font-weight: 600; }}

        /* Tables and lists */
        QTableView, QTableWidget, QListView, QListWidget, QTreeView, QTreeWidget {{
            background-color: {t.background_secondary}; alternate-background-color: {t.background_primary};
            border: 1px solid {t.border_color}; border-radius: {r}px; gridline-color: transparent; outline: none;
            selection-background-color: {t.accent_soft}; selection-color: {t.text_primary}; }}
        QTableView::item, QListView::item, QTreeView::item {{ padding: 6px 8px; border: none; }}
        QTableView::item:selected, QListView::item:selected, QTreeView::item:selected {{
            background-color: {t.accent_soft}; color: {t.text_primary}; }}
        QTableView::item:hover, QListView::item:hover {{ background-color: {t.hover}; }}
        QHeaderView {{ background: transparent; }}
        QHeaderView::section {{ background-color: {t.background_primary}; color: {t.text_secondary}; font-weight: 600;
                               padding: 8px 10px; border: none; border-bottom: 1px solid {t.border_color}; }}
        QTableCornerButton::section {{ background: {t.background_primary}; border: none; }}

        /* Scroll bars: slim */
        QScrollArea {{ border: none; background: transparent; }}
        QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
        QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {t.border_color}; border-radius: 4px; min-height: 28px; }}
        QScrollBar::handle:horizontal {{ background: {t.border_color}; border-radius: 4px; min-width: 28px; }}
        QScrollBar::handle:hover {{ background: {t.text_disabled}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

        /* Menus */
        QMenuBar {{ background: {t.background_secondary}; border-bottom: 1px solid {t.border_color}; }}
        QMenuBar::item {{ padding: 6px 12px; background: transparent; border-radius: 4px; }}
        QMenuBar::item:selected {{ background: {t.hover}; }}
        QMenu {{ background-color: {t.background_secondary}; border: 1px solid {t.border_color}; border-radius: {r}px; padding: 5px; }}
        QMenu::item {{ padding: 7px 22px 7px 14px; border-radius: 4px; }}
        QMenu::item:selected {{ background-color: {t.accent_soft}; color: {t.text_primary}; }}
        QMenu::item:disabled {{ color: {t.text_disabled}; }}
        QMenu::separator {{ height: 1px; background: {t.border_color}; margin: 5px 8px; }}

        /* Progress */
        QProgressBar {{ background: {t.background_tertiary}; border: none; border-radius: 4px; height: 8px;
                       text-align: center; color: transparent; }}
        QProgressBar::chunk {{ background: {t.accent_primary}; border-radius: 4px; }}

        /* Status bar */
        QStatusBar {{ background: {t.background_secondary}; color: {t.text_secondary}; border-top: 1px solid {t.border_color}; }}
        QStatusBar::item {{ border: none; }}
        QMessageBox {{ background: {t.background_primary}; }}
        """
        css = "\n".join(line[8:] if line.startswith("        ") else line for line in css.splitlines())
        self.cache[key] = css
        return css

    def clear_cache(self):
        self.cache.clear()
