from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class FollowUpPanel(QWidget):
    """Panel for displaying follow-up questions"""
    
    question_clicked = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup panel UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Follow-up Questions")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Questions container
        self.questions_layout = QVBoxLayout()
        layout.addLayout(self.questions_layout)
        
        # Placeholder
        self.placeholder = QLabel("No follow-up questions available")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #999; padding: 20px;")
        self.questions_layout.addWidget(self.placeholder)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_questions(self, questions: list):
        """Set follow-up questions"""
        # Clear existing questions
        self._clear_questions()
        
        if not questions:
            self.questions_layout.addWidget(self.placeholder)
            return
        
        for question in questions:
            self._add_question(question)
    
    def _add_question(self, question: str):
        """Add a single question"""
        btn = QPushButton(question)
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """)
        btn.clicked.connect(lambda: self.question_clicked.emit(question))
        self.questions_layout.addWidget(btn)
    
    def _clear_questions(self):
        """Clear all questions"""
        while self.questions_layout.count():
            item = self.questions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def clear(self):
        """Clear panel"""
        self._clear_questions()
        self.questions_layout.addWidget(self.placeholder)