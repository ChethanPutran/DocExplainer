from PySide6.QtWidgets import QStatusBar, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer


class StatusBar(QStatusBar):
    """Custom status bar with progress and message support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(30)
        self._setup_ui()
        
        # Message timer
        self.message_timer = QTimer()
        self.message_timer.setSingleShot(True)
        self.message_timer.timeout.connect(self._clear_temporary_message)
    
    def _setup_ui(self):
        """Setup status bar UI"""
        # Permanent widgets (right side)
        self.page_label = QLabel()
        self.page_label.setMinimumWidth(100)
        self.page_label.setAlignment(Qt.AlignRight)
        self.addPermanentWidget(self.page_label)
        
        self.position_label = QLabel()
        self.position_label.setMinimumWidth(100)
        self.position_label.setAlignment(Qt.AlignRight)
        self.addPermanentWidget(self.position_label)
        
        self.zoom_label = QLabel()
        self.zoom_label.setMinimumWidth(60)
        self.zoom_label.setAlignment(Qt.AlignRight)
        self.addPermanentWidget(self.zoom_label)
        
        self.doc_type_label = QLabel()
        self.doc_type_label.setMinimumWidth(60)
        self.doc_type_label.setAlignment(Qt.AlignRight)
        self.addPermanentWidget(self.doc_type_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setVisible(False)
        self.addPermanentWidget(self.progress_bar)
    
    def show_message(self, message: str, timeout: int = 3000):
        """Show temporary message"""
        self.showMessage(message, timeout)
    
    def show_permanent_message(self, message: str):
        """Show permanent message in status bar"""
        self.clearMessage()
        self.showMessage(message, 0)  # 0 = permanent
    
    def set_page_info(self, page: int, total_pages: int = None):
        """Set page information"""
        if total_pages:
            self.page_label.setText(f"Page {page}/{total_pages}")
        else:
            self.page_label.setText(f"Page {page}")
    
    def set_position_info(self, position: int, total_length: int = None):
        """Set position information"""
        if total_length:
            percentage = (position / total_length) * 100
            self.position_label.setText(f"Pos {percentage:.1f}%")
        else:
            self.position_label.setText(f"Pos {position}")
    
    def set_zoom_info(self, zoom: float):
        """Set zoom information"""
        self.zoom_label.setText(f"{zoom:.0%}")
    
    def set_doc_type(self, doc_type: str):
        """Set document type"""
        self.doc_type_label.setText(doc_type.upper())
    
    def show_progress(self, minimum: int = 0, maximum: int = 100):
        """Show progress bar"""
        self.progress_bar.setMinimum(minimum)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
    
    def update_progress(self, value: int):
        """Update progress value"""
        self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
    
    def show_ready(self):
        """Show ready state"""
        self.show_message("Ready", 2000)
        self.page_label.setText("")
        self.position_label.setText("")
    
    def show_loading(self, filename: str):
        """Show loading state"""
        self.show_permanent_message(f"Loading {filename}...")
        self.show_progress()
    
    def show_success(self, message: str = "Complete"):
        """Show success state"""
        self.show_message(f"✓ {message}", 3000)
        self.hide_progress()
    
    def show_error(self, error: str):
        """Show error state"""
        self.show_message(f"✗ Error: {error}", 5000)
        self.hide_progress()
    
    def show_warning(self, warning: str):
        """Show warning state"""
        self.show_message(f"⚠ {warning}", 4000)
    
    def show_info(self, info: str):
        """Show info state"""
        self.show_message(f"ℹ {info}", 3000)
    
    def _clear_temporary_message(self):
        """Clear temporary message"""
        self.clearMessage()