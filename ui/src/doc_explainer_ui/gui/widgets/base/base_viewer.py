from abc import abstractmethod
from typing import Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal
from PySide6.QtWidgets import QInputDialog, QMenu
from .base_widget import BaseWidget

class BaseDocumentViewer(BaseWidget):
    """Base class for document viewers"""
    doc_id: str
    # Signal emitted when text is selected
    text_selected = Signal(str, str, str, int, int)  # action, doc_id, text, page, position
    # Signal emitted when document is loaded
    document_loaded = Signal(str)  # doc_id
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.doc_id = ""  # set once the service has finished processing the document
        self.current_path: str = ""
        self.current_page: int = 1
        self.current_position: int = 0
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load document from path"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear document and free resources"""
        pass
    
    def get_selected_text(self) -> str:
        """Get currently selected text - to be overridden"""
        return ""
    
    def get_current_page(self) -> int:
        """Get current page number"""
        return self.current_page
    
    def get_current_position(self) -> int:
        """Get current text position"""
        return self.current_position
    
    # ------------------------------------------------------------------ find
    def find(self, text: str, forward: bool = True, case_sensitive: bool = False) -> bool:
        """Select/show the next (or previous) match; wraps around. False when there is none."""
        return False

    def clear_find(self) -> None:
        """Remove any match highlighting."""

    # ------------------------------------------------------------------ zoom
    ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.25, 5.0, 1.1

    current_zoom: float = 1.0

    def set_zoom(self, zoom: float) -> None:
        """Apply an absolute zoom level - overridden by each viewer."""

    def zoom_to(self, zoom: float) -> float:
        zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom))
        if abs(zoom - self.current_zoom) > 1e-6:
            self.current_zoom = zoom
            self.set_zoom(zoom)
        return self.current_zoom

    def zoom_in(self) -> float:
        return self.zoom_to(self.current_zoom * self.ZOOM_STEP)

    def zoom_out(self) -> float:
        return self.zoom_to(self.current_zoom / self.ZOOM_STEP)

    def zoom_reset(self) -> float:
        return self.zoom_to(1.0)

    def _attach_zoom(self, widget) -> None:
        """Ctrl + mouse wheel over ``widget`` zooms (a plain wheel still scrolls)."""
        surface = widget.viewport() if hasattr(widget, "viewport") else widget
        self._zoom_filter = _WheelZoom(self)
        surface.installEventFilter(self._zoom_filter)

    # ------------------------------------------------------------------ right-click menu
    MAX_CONTEXT_CHARS = 4000        # a whole page can be long; the tutor only needs the passage

    ai_note: str = ""               # why the AI tutor is not available yet (set by the window)

    def _attach_context_menu(self, widget) -> None:
        """Right-clicking ``widget`` shows this viewer's menu."""
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        surface = widget.viewport() if hasattr(widget, "viewport") else widget
        widget.customContextMenuRequested.connect(lambda pos, s=surface: self._show_menu(s.mapToGlobal(pos), pos))

    def _show_menu(self, global_pos: QPoint, pos: QPoint) -> None:
        self._build_menu(pos).exec(global_pos)

    def _context_target(self, pos: QPoint) -> Tuple[str, int, str]:
        """(text to act on, its page, what to call it) for a right click at ``pos``. Viewers that have no
        selectable text (a PDF page) override this."""
        return self.get_selected_text().strip(), self.current_page, "selection"

    def _edit_actions(self, menu: QMenu) -> None:
        """Copy / Select all for viewers that show selectable text. Default: none."""

    def _build_menu(self, pos: QPoint) -> QMenu:
        """The options that apply right now. AI options exist only once the server has the document."""
        menu = QMenu(self)
        self._edit_actions(menu)
        text, page, what = self._context_target(pos)
        if menu.actions():
            menu.addSeparator()
        ready = bool(self.doc_id)
        emit = lambda signal, t: signal.emit(self.doc_id, t[: self.MAX_CONTEXT_CHARS], page, self.current_position)  # noqa: E731
        explain = menu.addAction(f"Explain this {what}")
        summarize = menu.addAction(f"Summarize this {what}")
        ask = menu.addAction(f"Ask a question about this {what}...")
        explain.setEnabled(ready and bool(text))          # AI actions: only once the server has the document
        summarize.setEnabled(ready and bool(text))
        ask.setEnabled(ready)
        if ready and not text:
            explain.setToolTip("Select some text first")
            summarize.setToolTip("Select some text first")
        menu.addSeparator()
        save = menu.addAction(f"Save this {what} to notes")
        save.setEnabled(ready and bool(text))
        save.triggered.connect(lambda: self.signals.note_requested.emit(
            self.doc_id, f"\u201c{' '.join(text.split())[: self.MAX_CONTEXT_CHARS]}\u201d  (page {page})"))
        mark = menu.addAction("Bookmark this place")
        mark.setEnabled(ready)
        mark.triggered.connect(lambda: self.signals.note_requested.emit(self.doc_id, self.bookmark_text(page, text)))
        if hasattr(self, "highlight_selection"):
            hi = menu.addAction("Highlight selection")
            hi.setEnabled(bool(self.get_selected_text().strip()))
            hi.triggered.connect(self.highlight_selection)
            menu.addAction("Clear highlights").triggered.connect(self.clear_highlights)
        if not ready:                                     # everything else stays usable; say why these are grey
            menu.addSeparator()
            note = self.ai_note.split("\n")[0] if self.ai_note else "Document is being processed..."
            menu.addAction("\u23f3 " + note + "  (AI tutor available when ready)").setEnabled(False)
        explain.triggered.connect(lambda: emit(self.signals.explain_requested, text))
        summarize.triggered.connect(lambda: emit(self.signals.summarize_requested, text))
        ask.triggered.connect(lambda: self._ask(text, page))
        return menu

    def bookmark_text(self, page: int, text: str = "") -> str:
        snippet = " ".join(text.split())[:80]
        return f"\U0001F516 Bookmark: page {page}" + (f" \u2014 {snippet}" if snippet else "")

    def _ask(self, context: str, page: int) -> None:
        question, ok = QInputDialog.getText(self, "Ask the tutor", "Your question:")
        question = question.strip()
        if not ok or not question:
            return
        if context:
            question = f"{question}\n\nAbout this passage: \u201c{context[: self.MAX_CONTEXT_CHARS]}\u201d"
        self.signals.ask_requested.emit(self.doc_id, question, page, self.current_position)

    def go_to_section(self, page: int, title: str) -> bool:
        """Scroll to a section's heading (``title``) on ``page``. The default only knows the page."""
        return self.go_to_page(page)

    def section_at_position(self, sections) -> "Optional[int]":
        """For viewers without pages: the id of the section the reader is in, from ``(id, title)`` pairs. None = use the page."""
        return None

    def go_to_page(self, page: int) -> bool:
        """Scroll to a 1-based page. Viewers without pages return False."""
        return False

    def set_doc_id(self, doc_id: str):
        """Set document ID"""
        self.doc_id = doc_id
    
    def emit_text_action(self, action: str, text: str):
        """Emit text selection signal"""
        self.text_selected.emit(
            action,
            self.doc_id,
            text,
            self.current_page,
            self.current_position
        )


class _WheelZoom(QObject):
    """Turns Ctrl + wheel into zoom steps and swallows the event so the view does not also scroll."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self._viewer = viewer

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            dy = event.angleDelta().y()
            if dy > 0:
                self._viewer.zoom_in()
            elif dy < 0:
                self._viewer.zoom_out()
            return True
        return False
