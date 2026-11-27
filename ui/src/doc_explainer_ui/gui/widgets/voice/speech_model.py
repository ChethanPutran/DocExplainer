"""A process-wide, resident Whisper model: loaded once, kept in memory, unloaded when idle.

Loading Whisper takes seconds and ~150 MB, so it must not happen on every "start listening". This
cache loads it in the background on first need and keeps it; it is released only after
``idle_unload_s`` seconds with nobody listening or transcribing (0 = never release). The next use
simply loads it again.

Framework-free on purpose (no Qt, no microphone): it can be tested on its own.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

UNLOADED, LOADING, READY = "unloaded", "loading", "ready"


class ModelLoadError(RuntimeError):
    """The model could not be loaded."""


def pick_device() -> str:
    """"cuda" only if this PyTorch build can actually run on the installed GPU, else "cpu".

    ``torch.cuda.is_available()`` is not enough: wheels ship kernels for specific GPU generations,
    and on an older card (e.g. compute capability 6.1) loading a model fails with "no kernel image
    is available". Set DOC_EXPLAINER_WHISPER_DEVICE (cpu/cuda) to force a choice.
    """
    forced = os.getenv("DOC_EXPLAINER_WHISPER_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        with warnings.catch_warnings():  # torch warns loudly about unsupported GPUs; we log once below
            warnings.simplefilter("ignore")
            if not torch.cuda.is_available():
                return "cpu"
            major, minor = torch.cuda.get_device_capability()
            archs = torch.cuda.get_arch_list()  # e.g. ['sm_75', 'sm_80', ..., 'compute_120']
        cc = major * 10 + minor
        has_binary = f"sm_{cc}" in archs
        has_ptx = any(a.startswith("compute_") and int(a[8:]) <= cc for a in archs)  # JIT-able
        if has_binary or has_ptx:
            return "cuda"
        logger.warning(
            "GPU compute capability %d.%d is not supported by this PyTorch build (supports: %s). "
            "Running voice input on the CPU.", major, minor, ", ".join(archs))
    except Exception:
        logger.warning("Could not use the GPU for voice input; using the CPU.", exc_info=True)
    return "cpu"


def _default_loader(name: str) -> Any:
    import whisper  # imported here: it pulls in torch, which should not slow down app start-up

    device = pick_device()
    started = time.monotonic()
    model = whisper.load_model(name, device=device)
    logger.info("Whisper '%s' loaded on %s in %.1fs", name, device, time.monotonic() - started)
    return model


class WhisperModelCache:
    def __init__(self, loader: Optional[Callable[[str], Any]] = None, idle_unload_s: float = 600.0,
                 clock: Callable[[], float] = time.monotonic, poll_s: Optional[float] = None):
        self._loader = loader or _default_loader
        self._clock = clock
        self._poll_s = poll_s
        self._idle = float(idle_unload_s)
        self._cv = threading.Condition()
        self._infer = threading.Lock()        # one transcription at a time on the shared model
        self._model: Any = None
        self._name: Optional[str] = None
        self._state = UNLOADED
        self._error: Optional[BaseException] = None
        self._gen = 0                         # ignores the result of a superseded load
        self._in_use = 0                      # transcriptions in flight
        self._sessions = 0                    # open listening sessions
        self._last_used = clock()
        self._watcher: Optional[threading.Thread] = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------ introspection
    @property
    def state(self) -> str:
        return self._state

    def is_ready(self, name: Optional[str] = None) -> bool:
        with self._cv:
            return self._state == READY and (name is None or name == self._name)

    # ------------------------------------------------------------------ configuration
    def configure(self, idle_unload_s: Optional[float] = None) -> None:
        if idle_unload_s is not None:
            self._idle = float(idle_unload_s)

    # ------------------------------------------------------------------ loading
    def start_loading(self, name: str) -> None:
        """Begin loading in the background. Returns at once; a no-op if already loaded or loading."""
        with self._cv:
            if self._name == name and self._state in (READY, LOADING):
                self._last_used = self._clock()
                return
            self._gen += 1
            gen = self._gen
            self._name, self._state, self._model, self._error = name, LOADING, None, None
        threading.Thread(target=self._load, args=(name, gen), name="whisper-load", daemon=True).start()

    def _load(self, name: str, gen: int) -> None:
        try:
            model = self._loader(name)
        except BaseException as e:  # noqa: BLE001 - handed to whoever is waiting
            logger.exception("Could not load Whisper '%s'", name)
            with self._cv:
                if gen == self._gen:
                    self._state, self._error = UNLOADED, e
                self._cv.notify_all()
            return
        with self._cv:
            if gen != self._gen:  # someone asked for a different model meanwhile: drop this one
                return
            self._model, self._state, self._last_used = model, READY, self._clock()
            self._cv.notify_all()
        self._ensure_watcher()

    def wait_ready(self, name: str, timeout: Optional[float] = None) -> None:
        """Block until ``name`` is loaded (starting the load if needed). Raises ``ModelLoadError``."""
        self.start_loading(name)
        deadline = None if timeout is None else self._clock() + timeout
        with self._cv:
            while True:
                if self._state == READY and self._name == name:
                    return
                if self._state == UNLOADED:
                    raise ModelLoadError(f"Could not load the speech model '{name}': "
                                         f"{self._error or 'it was unloaded'}") from self._error
                if self._state == READY:  # a different model won the race: ask for ours again
                    self.start_loading(name)
                    continue
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Speech model '{name}' did not load in time")
                self._cv.wait(timeout=remaining)

    # ------------------------------------------------------------------ use
    @contextmanager
    def use(self, name: str, timeout: Optional[float] = None) -> Iterator[Any]:
        """Hold the model for one transcription. Waits for it to load if necessary."""
        self.wait_ready(name, timeout)
        with self._cv:
            if not (self._state == READY and self._name == name):
                raise ModelLoadError(f"The speech model '{name}' was unloaded before it could be used")
            self._in_use += 1
        try:
            with self._infer:
                yield self._model
        finally:
            with self._cv:
                self._in_use -= 1
                self._last_used = self._clock()

    @contextmanager
    def session(self) -> Iterator[None]:
        """While open (the microphone is on), the model counts as in use and is never unloaded."""
        with self._cv:
            self._sessions += 1
        try:
            yield
        finally:
            with self._cv:
                self._sessions -= 1
                self._last_used = self._clock()

    # ------------------------------------------------------------------ idle unload
    def unload_if_idle(self) -> bool:
        """Release the model if nobody has used it for ``idle_unload_s``. Returns True if released."""
        with self._cv:
            if (self._state != READY or self._idle <= 0 or self._in_use or self._sessions
                    or self._clock() - self._last_used < self._idle):
                return False
            self._model, self._state = None, UNLOADED
            self._gen += 1
            name = self._name
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Unloaded Whisper '%s' after %.0fs idle", name, self._idle)
        return True

    def _ensure_watcher(self) -> None:
        with self._cv:
            if self._watcher is not None or self._closed.is_set():
                return
            self._watcher = threading.Thread(target=self._watch, name="whisper-idle", daemon=True)
            self._watcher.start()

    def _watch(self) -> None:
        while not self._closed.wait(self._poll_s or max(1.0, self._idle / 10 if self._idle > 0 else 5.0)):
            self.unload_if_idle()

    def close(self) -> None:
        self._closed.set()


_cache: Optional[WhisperModelCache] = None
_cache_lock = threading.Lock()


def get_model_cache() -> WhisperModelCache:
    """The one cache shared by every voice widget in this process."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = WhisperModelCache()
        return _cache
