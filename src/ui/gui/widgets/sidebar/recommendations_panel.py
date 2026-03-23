from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont


class RecommendationsPanel(QWidget):
    """Panel for displaying learning resources"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup panel UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Learning Resources")
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
        
        # Resources container
        self.resources_layout = QVBoxLayout()
        layout.addLayout(self.resources_layout)
        
        # Placeholder
        self.placeholder = QLabel("No resources available")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #999; padding: 20px;")
        self.resources_layout.addWidget(self.placeholder)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_resources(self, resources: list):
        """Set resources to display"""
        # Clear existing resources
        self._clear_resources()
        
        if not resources:
            self.resources_layout.addWidget(self.placeholder)
            return
        
        for resource in resources:
            self._add_resource(resource)
    
    def _add_resource(self, resource):
        """Add a single resource"""
        # Resource container
        container = QFrame()
        container.setFrameShape(QFrame.Shape.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 5px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout(container)
        
        # Title
        title = QLabel(resource.get('title', 'Resource'))
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        description = QLabel(resource.get('description', ''))
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; font-size: 10pt;")
        layout.addWidget(description)
        
        # Type and difficulty
        meta = QLabel(f"Type: {resource.get('type', 'unknown')} | "
                     f"Difficulty: {resource.get('difficulty', 'intermediate')}")
        meta.setStyleSheet("color: #999; font-size: 9pt;")
        layout.addWidget(meta)
        
        # Open button
        if resource.get('url'):
            btn = QPushButton("Open Resource")
            btn.clicked.connect(lambda: self._open_url(resource['url']))
            layout.addWidget(btn)
        
        self.resources_layout.addWidget(container)
    
    def _open_url(self, url: str):
        """Open URL in default browser"""
        QDesktopServices.openUrl(QUrl(url))
    
    def _clear_resources(self):
        """Clear all resources"""
        while self.resources_layout.count():
            item = self.resources_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def clear(self):
        """Clear panel"""
        self._clear_resources()
        self.resources_layout.addWidget(self.placeholder)