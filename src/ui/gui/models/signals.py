from PySide6.QtCore import QObject, Signal


class UISignals(QObject):
    """Centralized signals for UI communication"""
    
    # Document signals
    document_opened = Signal(str, str)  # doc_id, path
    document_closed = Signal(str)        # doc_id
    document_changed = Signal(str, int)  # doc_id, section_id
    
    # Selection signals
    text_selected = Signal(str, str, int, int)  # doc_id, text, page, position
    
    # Action signals
    explain_requested = Signal(str, str, int, int)  # doc_id, text, page, position
    summarize_requested = Signal(str, str, int, int)
    ask_requested = Signal(str, str, int, int)
    follow_up_requested = Signal(str, str, int)      # doc_id, question, section_id
    
    # Explanation signals
    explanation_received = Signal(object, int)  # explanation, section_id
    explanation_updated = Signal(object)        # explanation_view_model
    
    # Voice signals
    voice_input_received = Signal(str)          # text
    voice_output_started = Signal()
    voice_output_finished = Signal()
    
    # UI state signals
    sidebar_toggled = Signal(bool)               # visible
    theme_changed = Signal(str)                  # theme name