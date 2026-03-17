import speech_recognition as sr
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, QThread, QTimer
from PySide6.QtGui import QIcon

from ..base.base_widget import BaseWidget


class ListeningThread(QThread):
    """Thread for listening to voice input"""
    
    text_recognized = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.is_listening = False
    
    def run(self):
        """Run listening loop"""
        self.is_listening = True
        
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        text = self.recognizer.recognize_google(audio)
                        self.text_recognized.emit(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        self.error_occurred.emit("Could not understand audio")
                    except Exception as e:
                        self.error_occurred.emit(str(e))
                        
        except Exception as e:
            self.error_occurred.emit(f"Microphone error: {e}")
    
    def stop(self):
        """Stop listening"""
        self.is_listening = False


class VoiceInput(BaseWidget):
    """Voice input widget"""
    
    voice_text = Signal(str)
    
    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)
        self.listening_thread = None
        self.is_recording = False
        
    def _setup_ui(self):
        """Setup voice input UI"""
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Click to start voice input")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.status_label)
        
        self.record_button = QPushButton("🎤 Start Listening")
        self.record_button.clicked.connect(self.toggle_listening)
        layout.addWidget(self.record_button)
        
        self.setLayout(layout)
    
    def _connect_signals(self):
        """Connect signals"""
        if self.signals:
            self.voice_text.connect(self.signals.voice_input_received)
    
    def toggle_listening(self):
        """Toggle voice input listening"""
        if not self.is_recording:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """Start voice input"""
        self.listening_thread = ListeningThread()
        self.listening_thread.text_recognized.connect(self._on_text_recognized)
        self.listening_thread.error_occurred.connect(self._on_error)
        self.listening_thread.start()
        
        self.is_recording = True
        self.record_button.setText("🔴 Stop Listening")
        self.status_label.setText("Listening...")
        self.status_label.setStyleSheet("color: #28a745; font-size: 9pt;")
    
    def stop_listening(self):
        """Stop voice input"""
        if self.listening_thread:
            self.listening_thread.stop()
            self.listening_thread.wait()
            self.listening_thread = None
        
        self.is_recording = False
        self.record_button.setText("🎤 Start Listening")
        self.status_label.setText("Click to start voice input")
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
    
    def _on_text_recognized(self, text: str):
        """Handle recognized text"""
        self.voice_text.emit(text)
        self.status_label.setText(f"Recognized: {text[:30]}...")
    
    def _on_error(self, error: str):
        """Handle error"""
        self.status_label.setText(f"Error: {error}")
        self.status_label.setStyleSheet("color: #dc3545; font-size: 9pt;")
        QTimer.singleShot(3000, self._reset_status)
    
    def _reset_status(self):
        """Reset status label"""
        if not self.is_recording:
            self.status_label.setText("Click to start voice input")
            self.status_label.setStyleSheet("color: #666; font-size: 9pt;")