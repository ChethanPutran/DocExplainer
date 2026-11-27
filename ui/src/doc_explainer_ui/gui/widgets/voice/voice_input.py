
import logging
import queue
import tempfile
import threading
import warnings

import speech_recognition as sr
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, QThread, Qt, QTimer

from ..base.base_widget import BaseWidget
from ...utils.audio_noise import silence_audio_library_noise
from .speech_model import ModelLoadError, get_model_cache, pick_device  # noqa: F401  (pick_device re-exported)

logger = logging.getLogger(__name__)

# Must run before PyAudio/PortAudio opens a device: it is what prints the ALSA/JACK noise.
silence_audio_library_noise()

# We choose the CPU on purpose when the GPU is unusable; whisper's reminder is then just noise.
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")


def audio_stats(audio) -> tuple:
    """(seconds, rms, peak) of a speech_recognition AudioData (16-bit samples).

    rms/peak are on a 0..32767 scale: a few dozen means a silent or muted microphone, a few
    hundred to thousands is normal speech.
    """
    seconds = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
    try:
        import numpy as np
        samples = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float64)
        rms, peak = float(np.sqrt(np.mean(samples ** 2))), float(np.max(np.abs(samples)))
    except Exception:
        rms = peak = float("nan")
    return seconds, rms, peak


class ListeningThread(QThread):
    """Listens on the microphone and turns speech into text with the shared Whisper model.

    Capturing and transcribing are separate, so nothing you say is lost:

    * the microphone opens at once, while the model may still be loading (``cache.start_loading``
      returns immediately, and is a no-op when the model is already resident);
    * every phrase goes into a queue; a transcriber thread works through it as soon as the model is
      ready, in order, while the microphone keeps listening;
    * stopping ends the capture within about a second. Phrases already spoken are still
      transcribed and delivered afterwards.
    """

    text_recognized = Signal(str)
    error_occurred = Signal(str)
    listening_started = Signal()   # the microphone is open and audio is being captured
    notice = Signal(str)           # something the user can fix (e.g. the microphone is far too loud)
    model_state = Signal(str)      # "loading" while the model is not yet usable, then "ready"

    NOISY_THRESHOLD = 1200         # calibrated ambient level above this: input gain is too high
    MAX_PENDING = 20               # phrases waiting for the model; the oldest is dropped beyond this

    def __init__(self, model_name="base", cache=None):
        super().__init__()

        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.model_name = model_name
        self._cache = cache or get_model_cache()

        logger.info("ListeningThread initialized with Whisper model=%s", model_name)

    # ------------------------------------------------------------------ capture (this QThread)
    def run(self):
        """Capture phrases until stopped."""
        self.is_listening = True
        pending: "queue.Queue" = queue.Queue()
        worker = None

        try:
            self._cache.start_loading(self.model_name)  # instant if resident; background otherwise
            self.model_state.emit("ready" if self._cache.is_ready(self.model_name) else "loading")

            with self._cache.session():  # the microphone being on counts as activity: no idle unload
                worker = threading.Thread(target=self._transcribe_loop, args=(pending,),
                                          name="doc-explainer-transcribe", daemon=True)
                worker.start()

                logger.info("Initializing microphone...")
                with sr.Microphone() as source:
                    logger.info("Microphone initialized (sample_rate=%s, chunk_size=%s)",
                                source.SAMPLE_RATE, source.CHUNK)
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    logger.info("Ambient noise calibration complete (energy_threshold=%s)",
                                self.recognizer.energy_threshold)

                    if self.recognizer.energy_threshold > self.NOISY_THRESHOLD:
                        logger.warning("Background level is very high (energy_threshold=%.0f): the microphone input "
                                       "gain is probably too high, so speech is hard to tell from noise",
                                       self.recognizer.energy_threshold)
                        self.notice.emit("Microphone is very loud/noisy: lower the input volume in system sound settings")
                    self.listening_started.emit()
                    logger.info("Voice listening loop started")

                    while self.is_listening:
                        try:
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        except sr.WaitTimeoutError:
                            continue
                        seconds, rms, peak = audio_stats(audio)
                        logger.info("Audio captured (duration=%.2fs, level rms=%.0f peak=%.0f)",
                                    seconds, rms, peak)
                        self._enqueue(pending, audio)

        except Exception as e:
            logger.exception("Voice input initialization failed")
            self.error_occurred.emit(f"Microphone error: {e}")

        finally:
            self.is_listening = False
            pending.put(None)  # after the phrases already captured: let the transcriber finish them
            logger.info("ListeningThread stopped capturing")

    def _enqueue(self, pending: "queue.Queue", audio) -> None:
        while pending.qsize() >= self.MAX_PENDING:
            try:
                pending.get_nowait()
                logger.warning("Speech model is still not ready; dropped the oldest waiting phrase")
            except queue.Empty:
                break
        pending.put(audio)

    # ------------------------------------------------------------------ transcription (own thread)
    def _transcribe_loop(self, pending: "queue.Queue") -> None:
        try:
            self._cache.wait_ready(self.model_name)  # capture keeps going meanwhile
        except ModelLoadError as e:
            logger.error("%s", e)
            self.error_occurred.emit(str(e))
            self.is_listening = False
            return
        self.model_state.emit("ready")

        while True:
            audio = pending.get()
            if audio is None:
                return
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    f.write(audio.get_wav_data())
                    f.flush()
                    with self._cache.use(self.model_name) as model:
                        result = model.transcribe(f.name, fp16=False)
                text = result["text"].strip()
                if text:
                    logger.info("Speech recognized: %s", text)
                    self.text_recognized.emit(text)
                else:
                    logger.info("Whisper heard no speech in that clip "
                                "(if the level above is very low, check the microphone)")
            except ModelLoadError as e:
                logger.error("%s", e)
                self.error_occurred.emit(str(e))
            except Exception:
                logger.exception("Error during speech recognition")
                self.error_occurred.emit("Speech recognition error")

    def stop(self):
        """Stop capturing. Returns at once; already-captured phrases are still transcribed."""
        logger.info("Stopping ListeningThread")
        self.is_listening = False


class VoiceInput(BaseWidget):
    """Voice input widget."""

    voice_text = Signal(str)
    recording_changed = Signal(bool)     # the microphone was switched on or off (also when it failed)
    status_changed = Signal(str)         # what the microphone is doing, for whoever shows it

    def __init__(self, parent=None, signals=None, voice_config=None):
        self._voice_config = voice_config
        self.listening_thread = None
        self.is_recording = False
        self._mic_ready = False
        self._model_ready = False
        self._finishing = []  # stopped threads that are still delivering the last phrases
        super().__init__(parent, signals)

        cache = get_model_cache()
        if voice_config is not None:
            cache.configure(idle_unload_s=voice_config.model_idle_unload_s)
            if voice_config.preload_model:  # load in the background now: even the first use is instant
                cache.start_loading(voice_config.whisper_model)

        logger.info("VoiceInput widget initialized")

    def _model_name(self) -> str:
        return self._voice_config.whisper_model if self._voice_config else "base"

    def _setup_ui(self):
        """Setup voice input UI."""
        layout = QVBoxLayout()
        self.status_label = QLabel("Click to start voice input")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.status_label)

        self.record_button = QPushButton("🎤 Start Listening")
        self.record_button.clicked.connect(self.toggle_listening)
        layout.addWidget(self.record_button)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect signals."""
        if self.signals:
            self.voice_text.connect(self.signals.voice_input_received)

    def toggle_listening(self):
        """Toggle voice input listening."""
        logger.info("Voice input toggle requested (currently_recording=%s)", self.is_recording)
        if not self.is_recording:
            self.start_listening()
        else:
            self.stop_listening()

    # ------------------------------------------------------------------ status
    def _set_status(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {colour}; font-size: 9pt;")
        self.status_changed.emit(text)

    def _refresh_status(self) -> None:
        """Say "Listening" only once audio is really being captured."""
        if not self.is_recording:
            return
        if not self._mic_ready:
            self._set_status("Starting microphone...", "#b8860b")
        elif not self._model_ready:
            self._set_status("Listening... (loading the speech model; what you say is kept)", "#28a745")
        else:
            self._set_status("Listening...", "#28a745")

    def _on_listening_started(self):
        self._mic_ready = True
        self._refresh_status()

    def _on_model_state(self, state: str):
        self._model_ready = state == "ready"
        self._refresh_status()

    # ------------------------------------------------------------------ start / stop
    def start_listening(self):
        """Start voice input."""
        if self.is_recording:
            logger.warning("start_listening called while already recording")
            return

        logger.info("Starting voice input")
        try:
            thread = ListeningThread(self._model_name())
            thread.text_recognized.connect(self._on_text_recognized)
            thread.error_occurred.connect(self._on_error)
            thread.listening_started.connect(self._on_listening_started)
            thread.model_state.connect(self._on_model_state)
            thread.notice.connect(lambda text: self._set_status(text, "#b8860b"))
            thread.finished.connect(self._prune_finished)

            self.listening_thread = thread
            self.is_recording = True
            self._mic_ready = False
            self._model_ready = get_model_cache().is_ready(self._model_name())
            self.record_button.setText("🔴 Stop Listening")
            self._refresh_status()  # "Starting microphone..." until the thread says it is capturing
            thread.start()
            self.recording_changed.emit(True)
        except Exception:
            logger.exception("Failed to start voice input")
            self.is_recording = False
            self.listening_thread = None
            self.recording_changed.emit(False)
            self._on_error("Failed to start voice input")

    def stop_listening(self):
        """Stop voice input. Never blocks the UI: the thread ends on its own within about a second."""
        logger.info("Stopping voice input")
        thread, self.listening_thread = self.listening_thread, None
        if thread:
            thread.stop()
            self._finishing.append(thread)  # keep it alive until it has finished

        self.is_recording = False
        self._mic_ready = False
        self.record_button.setText("🎤 Start Listening")
        self._set_status("Click to start voice input", "#666")
        self.recording_changed.emit(False)

    def _prune_finished(self):
        self._finishing = [t for t in self._finishing if not t.isFinished()]

    def _on_text_recognized(self, text: str):
        """Handle recognized text."""
        logger.info("Voice text received: %s", text)
        self.voice_text.emit(text)
        self.status_label.setText(f"Recognized: {text[:30]}...")

    def _on_error(self, error: str):
        """Handle error."""
        logger.error("Voice input error: %s", error)
        if self.is_recording:
            self.stop_listening()          # a broken microphone: switch it off so the button shows the truth
        self._set_status(f"Error: {error}", "#dc3545")
        QTimer.singleShot(3000, self._reset_status)

    def _reset_status(self):
        """Reset status label."""
        if not self.is_recording:
            self._set_status("Click to start voice input", "#666")
