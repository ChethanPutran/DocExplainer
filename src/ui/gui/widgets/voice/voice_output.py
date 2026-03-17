import pyttsx3
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, QThread

from ..base.base_widget import BaseWidget


class SpeechThread(QThread):
    """Thread for text-to-speech"""
    
    started = Signal()
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init()
        self.text = ""
        self.is_speaking = False
    
    def set_text(self, text: str):
        """Set text to speak"""
        self.text = text
    
    def run(self):
        """Run speech synthesis"""
        self.is_speaking = True
        self.started.emit()
        
        self.engine.say(self.text)
        self.engine.runAndWait()
        
        self.is_speaking = False
        self.finished.emit()
    
    def stop(self):
        """Stop speaking"""
        if self.is_speaking:
            self.engine.stop()
            self.is_speaking = False


class VoiceOutput(BaseWidget):
    """Voice output widget"""
    
    tts_started = Signal()
    tts_finished = Signal()
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.speech_thread = None
        self.is_speaking = False
        
    def _setup_ui(self):
        """Setup voice output UI"""
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.status_label)
        
        self.speak_button = QPushButton("🔊 Speak")
        self.speak_button.clicked.connect(self.toggle_speech)
        self.speak_button.setEnabled(False)
        layout.addWidget(self.speak_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_speaking)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        self.setLayout(layout)
    
    def set_text(self, text: str):
        """Set text to speak"""
        self.current_text = text
        self.speak_button.setEnabled(True)
    
    def toggle_speech(self):
        """Toggle speech playback"""
        if not self.is_speaking:
            self.speak()
        else:
            self.stop_speaking()
    
    def speak(self, text: str = None):
        """Start speaking"""
        if text:
            self.current_text = text
        
        if not self.current_text:
            return
        
        self.speech_thread = SpeechThread()
        self.speech_thread.set_text(self.current_text)
        self.speech_thread.started.connect(self._on_speech_started)
        self.speech_thread.finished.connect(self._on_speech_finished)
        self.speech_thread.start()
        
        if self.signals:
            self.signals.voice_output_started.emit()
        
        self.tts_started.emit()
    
    def stop_speaking(self):
        """Stop speaking"""
        if self.speech_thread and self.speech_thread.is_speaking:
            self.speech_thread.stop()
            self.speech_thread.wait()
            self._on_speech_finished()
    
    def _on_speech_started(self):
        """Handle speech started"""
        self.is_speaking = True
        self.speak_button.setText("🔊 Speaking...")
        self.speak_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Speaking...")
        self.status_label.setStyleSheet("color: #28a745; font-size: 9pt;")
    
    def _on_speech_finished(self):
        """Handle speech finished"""
        self.is_speaking = False
        self.speak_button.setText("🔊 Speak")
        self.speak_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        
        if self.signals:
            self.signals.voice_output_finished.emit()
        
        self.tts_finished.emit()