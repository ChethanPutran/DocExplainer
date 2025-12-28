from PySide6.QtWidgets import QPushButton, QMessageBox
from PySide6.QtCore import Signal
import pyttsx3

class VoiceOutput(QPushButton):
    # Optional: emit when speech starts or finishes
    tts_started = Signal()
    tts_finished = Signal()

    def __init__(self, parent=None):
        super().__init__("🔊 Read Explanation", parent)
        self.clicked.connect(self.play_voice)
        self.engine = pyttsx3.init()
        self.text_to_speak = "This is a placeholder explanation."  # default

    def set_text(self, text: str):
        """Set the text that will be spoken"""
        self.text_to_speak = text

    def play_voice(self):
        """Play TTS"""
        print("VoiceOutput: Starting TTS playback.")
        print(f"VoiceOutput: Text to speak: {self.text_to_speak}")
        if not self.text_to_speak.strip():
            QMessageBox.warning(self, "Voice Output", "No text to speak.")
            return

        self.tts_started.emit()
        try:
            self.engine.say(self.text_to_speak)
            self.engine.runAndWait()
            self.tts_finished.emit()
        except Exception as e:
            QMessageBox.critical(self, "Voice Output", f"TTS error: {e}")
