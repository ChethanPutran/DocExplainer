"""Piper: a free, offline neural voice. It looks like a pyttsx3 engine (say / runAndWait / stop / setProperty) so
``SentenceSpeaker`` can use either. ``engine_factory`` picks Piper when it is installed and its voice file exists,
and quietly falls back to the system voice otherwise."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)
BASE_WORDS_PER_MINUTE = 150          # the configured ``output_rate`` is in words/minute; this one is Piper's natural pace


class PiperEngine:
    def __init__(self, model_path: Path):
        from piper import PiperVoice
        self._voice = PiperVoice.load(str(model_path))
        self._text = ""
        self._length_scale = 1.0
        self._volume = 1.0
        self._stopped = threading.Event()
        self._stream = None

    def setProperty(self, name: str, value) -> None:
        if name == "rate" and value:
            self._length_scale = max(0.6, min(1.6, BASE_WORDS_PER_MINUTE / float(value)))
        elif name == "volume":
            self._volume = max(0.0, min(1.0, float(value)))

    def say(self, text: str) -> None:
        self._text = text

    def runAndWait(self) -> None:
        import numpy as np
        import sounddevice as sd
        from piper import SynthesisConfig

        text, self._text = self._text, ""
        self._stopped.clear()
        cfg = SynthesisConfig(length_scale=self._length_scale, volume=self._volume)
        stream = None
        try:
            for chunk in self._voice.synthesize(text, cfg):
                if self._stopped.is_set():
                    return
                if stream is None:
                    stream = sd.OutputStream(samplerate=self._voice.config.sample_rate, channels=1, dtype="int16")
                    self._stream = stream
                    stream.start()
                stream.write(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).reshape(-1, 1))
        finally:
            self._stream = None
            if stream is not None:
                try:
                    stream.abort() if self._stopped.is_set() else stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Closing the audio stream failed", exc_info=True)

    def stop(self) -> None:
        self._stopped.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:  # noqa: BLE001
                pass


def _system_engine():
    import pyttsx3
    return pyttsx3.init()


def engine_factory(voice) -> Callable:
    """``voice`` is the UI's VoiceConfig."""
    choice = (getattr(voice, "output_engine", "auto") or "auto").lower()

    def make():
        if choice in ("auto", "piper"):
            model = Path(voice.piper_voices_dir).expanduser() / f"{voice.piper_voice}.onnx"
            try:
                if not model.exists():
                    raise FileNotFoundError(f"Piper voice not found: {model}")
                return PiperEngine(model)
            except Exception as e:  # noqa: BLE001
                if choice == "piper":
                    logger.warning("Piper unavailable (%s); using the system voice", e)
                else:
                    logger.info("Piper not used (%s); using the system voice", e)
        return _system_engine()

    return make
