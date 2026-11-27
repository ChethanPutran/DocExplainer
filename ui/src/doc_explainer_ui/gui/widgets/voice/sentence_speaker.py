"""Reads a reply aloud one sentence at a time and says when each sentence STARTS being spoken.

That is what lets the text on screen be revealed in step with the voice (a live transcript). One worker thread
owns the speech engine for its whole life; sentences queue up while it speaks, and ``stop`` empties the queue and
interrupts the sentence in progress. If no speech engine can be started the speaker reports ``unavailable`` once
and everything else carries on silently.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from .speech_text import speakable

logger = logging.getLogger(__name__)


def default_engine():
    import pyttsx3
    return pyttsx3.init()


class SentenceSpeaker(QObject):
    sentence_started = Signal(str)       # this sentence is now being spoken
    idle = Signal()                      # everything queued has been spoken (or was stopped)
    unavailable = Signal(str)            # no speech engine: why

    def __init__(self, engine_factory: Callable = default_engine, rate: Optional[int] = None, volume: Optional[float] = None):
        super().__init__()
        self._factory, self._rate, self._volume = engine_factory, rate, volume
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._pending = 0               # sentences queued or being spoken
        self._epoch = 0                 # bumped by stop(): sentences from before it are dropped
        self._failed = False

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._pending > 0

    def enqueue(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence or self._failed:
            return
        with self._lock:
            self._pending += 1
            epoch = self._epoch
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="doc-explainer-speaker", daemon=True)
                self._thread.start()
        self._queue.put((epoch, sentence))

    def stop(self) -> None:
        """Drop what is queued and interrupt the sentence being spoken."""
        with self._lock:
            self._epoch += 1
            self._pending = 0
            engine = self._engine
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001 - interrupting is best effort
                logger.debug("Could not interrupt the speech engine", exc_info=True)
        self.idle.emit()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        try:
            engine = self._factory()
            if self._rate:
                engine.setProperty("rate", self._rate)
            if self._volume is not None:
                engine.setProperty("volume", self._volume)
            self._engine = engine
        except Exception as e:  # noqa: BLE001
            logger.warning("No speech engine available: %s", e)
            self._failed = True
            with self._lock:
                self._pending = 0
            self.unavailable.emit(str(e) or type(e).__name__)
            self.idle.emit()
            return
        while True:
            try:
                epoch, sentence = self._queue.get(timeout=2.0)
            except queue.Empty:
                with self._lock:
                    if self._pending == 0:        # nothing more to say: let this thread end (a new one starts on demand)
                        self._thread = None
                        return
                continue
            if epoch != self._epoch:
                continue
            self.sentence_started.emit(sentence)
            try:
                spoken = speakable(sentence)
                if spoken:
                    engine.say(spoken)
                    engine.runAndWait()
            except Exception:  # noqa: BLE001
                logger.warning("Speech failed for a sentence", exc_info=True)
            with self._lock:
                if epoch == self._epoch:
                    self._pending = max(0, self._pending - 1)
                    finished = self._pending == 0
                else:
                    finished = False
            if finished:
                self.idle.emit()
