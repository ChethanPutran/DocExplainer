from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QTabWidget, QTextEdit)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QFont, QDesktopServices


class AboutWindow(QDialog):
    """About dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Doc Explainer")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup about dialog UI"""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo placeholder
        logo_label = QLabel()
        logo_label.setFixedSize(64, 64)
        logo_label.setStyleSheet("background-color: #007bff; border-radius: 32px;")
        header_layout.addWidget(logo_label)
        
        # Title and version
        title_layout = QVBoxLayout()
        title = QLabel("Doc Explainer")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title_layout.addWidget(title)
        
        version = QLabel("Version 1.0.0")
        version.setStyleSheet("color: #666;")
        title_layout.addWidget(version)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Tab widget
        tabs = QTabWidget()
        
        # About tab
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
            <h2>Doc Explainer</h2>
            <p>An intelligent document explanation system that helps you understand 
            complex documents through AI-powered explanations, summarization, 
            and question answering.</p>
            
            <h3>Features</h3>
            <ul>
                <li>Document viewing (PDF, text, HTML)</li>
                <li>AI-powered explanations</li>
                <li>Text summarization</li>
                <li>Question answering</li>
                <li>Voice input/output</li>
                <li>Knowledge graph visualization</li>
                <li>Personalized learning paths</li>
            </ul>
            
            <h3>Technologies</h3>
            <ul>
                <li>PySide6 for GUI</li>
                <li>LangChain for AI orchestration</li>
                <li>Google Gemini for LLM</li>
                <li>FAISS for vector search</li>
                <li>NetworkX for knowledge graphs</li>
            </ul>
        """)
        tabs.addTab(about_text, "About")
        
        # License tab
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setPlainText("""
MIT License

Copyright (c) 2024 Doc Explainer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
        """)
        tabs.addTab(license_text, "License")
        
        # Credits tab
        credits_text = QTextEdit()
        credits_text.setReadOnly(True)
        credits_text.setHtml("""
            <h3>Development Team</h3>
            <p>Doc Explainer was created by a team of AI and UI/UX experts 
            to make document understanding accessible to everyone.</p>
            
            <h3>Open Source Libraries</h3>
            <ul>
                <li><b>PySide6</b> - Qt for Python</li>
                <li><b>LangChain</b> - LLM framework</li>
                <li><b>FAISS</b> - Vector similarity search</li>
                <li><b>NetworkX</b> - Graph algorithms</li>
                <li><b>spaCy</b> - NLP processing</li>
                <li><b>PyMuPDF</b> - PDF processing</li>
                <li><b>SpeechRecognition</b> - Voice input</li>
                <li><b>pyttsx3</b> - Text-to-speech</li>
            </ul>
            
            <h3>Contributing</h3>
            <p>Visit our GitHub repository to contribute or report issues.</p>
        """)
        tabs.addTab(credits_text, "Credits")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        github_btn = QPushButton("GitHub")
        github_btn.clicked.connect(self._open_github)
        button_layout.addWidget(github_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _open_github(self):
        """Open GitHub repository"""
        QDesktopServices.openUrl(QUrl("https://github.com/yourusername/doc-explainer"))