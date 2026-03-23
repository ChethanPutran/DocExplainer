from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ExplanationPanel(QWidget):
    """Panel for displaying explanations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup panel UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Explanation")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Explanation content in scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        self.explanation_label = QLabel("Select text to get an explanation")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setTextFormat(Qt.TextFormat.RichText)
        self.content_layout.addWidget(self.explanation_label)
        
        # Known concepts section
        self.known_label = QLabel("")
        self.known_label.setWordWrap(True)
        self.known_label.setStyleSheet("color: #28a745;")
        self.content_layout.addWidget(self.known_label)
        
        # Unknown concepts section
        self.unknown_label = QLabel("")
        self.unknown_label.setWordWrap(True)
        self.unknown_label.setStyleSheet("color: #dc3545;")
        self.content_layout.addWidget(self.unknown_label)
        
        self.content_layout.addStretch()
        self.scroll_area.setWidget(self.content_widget)
        
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
    
    def set_explanation(self, explanation: str, known_concepts: list, unknown_concepts: list):
        """Set explanation content"""
        self.explanation_label.setText(explanation)
        
        if known_concepts:
            self.known_label.setText(f"✅ Known: {', '.join(known_concepts)}")
        else:
            self.known_label.setText("")
        
        if unknown_concepts:
            self.unknown_label.setText(f"📚 Learning: {', '.join(unknown_concepts)}")
        else:
            self.unknown_label.setText("")
    
    def clear(self):
        """Clear panel"""
        self.explanation_label.setText("Select text to get an explanation")
        self.known_label.setText("")
        self.unknown_label.setText("")