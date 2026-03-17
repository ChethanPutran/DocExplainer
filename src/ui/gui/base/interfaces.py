from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from PySide6.QtCore import QObject, Signal


class DocumentViewerInterface(ABC):
    """Interface for document viewers"""
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load document from path"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear document and free resources"""
        pass
    
    @abstractmethod
    def get_selected_text(self) -> str:
        """Get currently selected text"""
        pass
    
    @abstractmethod
    def get_current_page(self) -> int:
        """Get current page number"""
        pass
    
    @abstractmethod
    def get_current_position(self) -> int:
        """Get current text position"""
        pass
    
    @abstractmethod
    def set_doc_id(self, doc_id: str):
        """Set document ID"""
        pass


class SidebarInterface(ABC):
    """Interface for sidebar widget"""
    
    @abstractmethod
    def update_explanation(self, explanation: Any, section_id: int):
        """Update explanation display"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear sidebar content"""
        pass


class VoiceInterface(ABC):
    """Interface for voice widgets"""
    
    @abstractmethod
    def start_listening(self):
        """Start voice input"""
        pass
    
    @abstractmethod
    def stop_listening(self):
        """Stop voice input"""
        pass
    
    @abstractmethod
    def speak(self, text: str):
        """Speak text"""
        pass