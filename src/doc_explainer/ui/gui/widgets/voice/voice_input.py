
import logging

import speech_recognition as sr
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, QThread, Qt, QTimer
import whisper
import tempfile
from ..base.base_widget import BaseWidget


logger = logging.getLogger(__name__)




class ListeningThread(QThread):
    """Thread for listening to voice input and transcribing with Whisper."""

    text_recognized = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_name="base"):
        super().__init__()

        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.model_name = model_name
        self.model = None

        logger.info(
            "ListeningThread initialized with Whisper model=%s",
            model_name,
        )

    def run(self):
        """Run listening loop."""
        self.is_listening = True

        try:
            # Load Whisper inside the worker thread so GUI startup
            # does not get blocked.
            logger.info(
                "Loading Whisper model '%s'...",
                self.model_name,
            )

            self.model = whisper.load_model(self.model_name)

            logger.info(
                "Whisper model '%s' loaded successfully",
                self.model_name,
            )

            logger.info("Initializing microphone...")

            with sr.Microphone() as source:

                logger.info(
                    "Microphone initialized successfully "
                    "(sample_rate=%s, chunk_size=%s)",
                    source.SAMPLE_RATE,
                    source.CHUNK,
                )

                logger.info(
                    "Adjusting microphone for ambient noise..."
                )

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5,
                )

                logger.info(
                    "Ambient noise calibration complete "
                    "(energy_threshold=%s)",
                    self.recognizer.energy_threshold,
                )

                logger.info("Voice listening loop started")

                while self.is_listening:

                    try:
                        logger.debug("Waiting for speech...")

                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=5,
                        )

                        logger.info(
                            "Audio captured successfully "
                            "(duration=%.2fs)",
                            len(audio.frame_data)
                            / audio.sample_rate,
                        )

                        with tempfile.NamedTemporaryFile(
                            suffix=".wav",
                            delete=True,
                        ) as f:
                            f.write(audio.get_wav_data())
                            f.flush()

                            result = self.model.transcribe(
                                f.name,
                                fp16=False,
                            )

                        text = result["text"].strip()

                        if text:
                            logger.info(
                                "Speech recognized: %s",
                                text,
                            )

                            self.text_recognized.emit(text)
                        else:
                            logger.debug(
                                "Whisper returned empty transcription"
                            )

                    except sr.WaitTimeoutError:
                        logger.debug(
                            "No speech detected within timeout"
                        )
                        continue

                    except Exception:
                        logger.exception(
                            "Error during speech recognition"
                        )

                        self.error_occurred.emit(
                            "Speech recognition error"
                        )

        except Exception as e:
            logger.exception(
                "Voice input initialization failed"
            )

            self.error_occurred.emit(
                f"Microphone/Whisper error: {e}"
            )

        finally:
            self.is_listening = False
            logger.info("ListeningThread stopped")

    def stop(self):
        """Stop listening."""
        logger.info("Stopping ListeningThread")
        self.is_listening = False




class VoiceInput(BaseWidget):
    """Voice input widget."""

    voice_text = Signal(str)

    def __init__(self, parent=None, signals=None):
        super().__init__(parent, signals)

        self.listening_thread = None
        self.is_recording = False

        logger.info("VoiceInput widget initialized")

    def _setup_ui(self):
        """Setup voice input UI."""
        logger.debug("Setting up VoiceInput UI")

        layout = QVBoxLayout()

        self.status_label = QLabel(
            "Click to start voice input"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.status_label.setStyleSheet(
            "color: #666; font-size: 9pt;"
        )

        layout.addWidget(self.status_label)

        self.record_button = QPushButton(
            "🎤 Start Listening"
        )

        self.record_button.clicked.connect(
            self.toggle_listening
        )

        layout.addWidget(self.record_button)

        self.setLayout(layout)

        logger.debug("VoiceInput UI setup complete")

    def _connect_signals(self):
        """Connect signals."""
        logger.debug("Connecting VoiceInput signals")

        if self.signals:
            self.voice_text.connect(
                self.signals.voice_input_received
            )

            logger.debug(
                "voice_text connected to voice_input_received"
            )
        else:
            logger.debug(
                "No external signals provided"
            )

    def toggle_listening(self):
        """Toggle voice input listening."""
        logger.info(
            "Voice input toggle requested "
            "(currently_recording=%s)",
            self.is_recording,
        )

        if not self.is_recording:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        """Start voice input."""
        if self.is_recording:
            logger.warning(
                "start_listening called while already recording"
            )
            return

        logger.info("Starting voice input")

        try:
            self.listening_thread = ListeningThread()

            self.listening_thread.text_recognized.connect(
                self._on_text_recognized
            )

            self.listening_thread.error_occurred.connect(
                self._on_error
            )

            self.listening_thread.start()

            self.is_recording = True

            self.record_button.setText(
                "🔴 Stop Listening"
            )

            self.status_label.setText(
                "Listening..."
            )

            self.status_label.setStyleSheet(
                "color: #28a745; font-size: 9pt;"
            )

            logger.info("Voice input started successfully")

        except Exception:
            logger.exception(
                "Failed to start voice input"
            )

            self.is_recording = False
            self.listening_thread = None

            self._on_error(
                "Failed to start voice input"
            )

    def stop_listening(self):
        """Stop voice input."""
        logger.info("Stopping voice input")

        if self.listening_thread:
            logger.debug("Stopping ListeningThread")

            self.listening_thread.stop()

            logger.debug("Waiting for ListeningThread to finish")

            self.listening_thread.wait()

            logger.debug("ListeningThread finished")

            self.listening_thread = None

        self.is_recording = False

        self.record_button.setText(
            "🎤 Start Listening"
        )

        self.status_label.setText(
            "Click to start voice input"
        )

        self.status_label.setStyleSheet(
            "color: #666; font-size: 9pt;"
        )

        logger.info("Voice input stopped successfully")

    def _on_text_recognized(self, text: str):
        """Handle recognized text."""
        logger.info(
            "Voice text received: %s",
            text,
        )

        self.voice_text.emit(text)

        self.status_label.setText(
            f"Recognized: {text[:30]}..."
        )

    def _on_error(self, error: str):
        """Handle error."""
        logger.error(
            "Voice input error: %s",
            error,
        )

        self.status_label.setText(
            f"Error: {error}"
        )

        self.status_label.setStyleSheet(
            "color: #dc3545; font-size: 9pt;"
        )

        QTimer.singleShot(
            3000,
            self._reset_status,
        )

    def _reset_status(self):
        """Reset status label."""
        if not self.is_recording:
            logger.debug("Resetting VoiceInput status")

            self.status_label.setText(
                "Click to start voice input"
            )

            self.status_label.setStyleSheet(
                "color: #666; font-size: 9pt;"
            )
