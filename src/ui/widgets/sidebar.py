from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit

class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.explanation_label = QLabel("Explanation:")
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)

        self.layout.addWidget(self.explanation_label)
        self.layout.addWidget(self.explanation_text)
        self.setLayout(self.layout)

    def update_explanation(self, text):
        self.explanation_text.setPlainText(text)
