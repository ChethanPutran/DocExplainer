from PySide6.QtWidgets import QApplication, QPushButton, QMessageBox
from PySide6.QtCore import Signal
import speech_recognition as sr

class VoiceInput(QPushButton):
    voice_text = Signal(str)

    def __init__(self, parent=None):
        super().__init__("🎤 Ask by Voice", parent)
        self.clicked.connect(self.record_voice)

    def record_voice(self):
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                self.setText("🎙 Listening...")
                QApplication.processEvents()  # <-- Fix: QApplication imported
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            text = recognizer.recognize_google(audio)
            self.voice_text.emit(text)
            self.setText("🎤 Ask by Voice")
            QMessageBox.information(self, "Voice Input", f"Recognized: {text}")

        except sr.WaitTimeoutError:
            self.setText("🎤 Ask by Voice")
            QMessageBox.warning(self, "Voice Input", "Listening timed out. Please try again.")
        except sr.UnknownValueError:
            self.setText("🎤 Ask by Voice")
            QMessageBox.warning(self, "Voice Input", "Could not understand audio.")
        except sr.RequestError as e:
            self.setText("🎤 Ask by Voice")
            QMessageBox.critical(self, "Voice Input", f"Could not request results; {e}")
