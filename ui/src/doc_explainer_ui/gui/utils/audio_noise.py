"""Silence the audio libraries' own console chatter.

Opening a microphone makes PortAudio probe every ALSA and JACK device, and those C libraries print
dozens of "ALSA lib ... Unknown PCM", "jack server is not running" lines straight to stderr. They are
not errors (the microphone works), and Python's logging/warnings filters cannot catch them. Both
libraries let you replace their message handlers, so that is what this does: precise, process-wide,
and it does not touch stderr, so real log lines from other threads are never lost.

Set DOC_EXPLAINER_AUDIO_DEBUG=1 to keep the messages when you are diagnosing a microphone problem.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

_installed = False
_keep_alive: list = []  # ctypes callbacks must outlive the libraries' use of them


def _load(*names: str):
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    return None


def silence_audio_library_noise() -> bool:
    """Install silent ALSA/JACK message handlers. Safe to call repeatedly; never raises.

    Returns True if at least one library was silenced.
    """
    global _installed
    if _installed or os.getenv("DOC_EXPLAINER_AUDIO_DEBUG"):
        return _installed
    silenced = False

    # ALSA: void handler(const char *file, int line, const char *function, int err, const char *fmt, ...)
    # The handler is variadic in C; ignoring every argument is safe.
    asound = _load("libasound.so.2", ctypes.util.find_library("asound") or "libasound.so")
    if asound is not None:
        try:
            alsa_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                                            ctypes.c_int, ctypes.c_char_p)(lambda *args: None)
            asound.snd_lib_error_set_handler(alsa_handler)
            _keep_alive.append(alsa_handler)
            silenced = True
        except Exception:
            logger.debug("Could not silence ALSA messages", exc_info=True)

    # JACK: void handler(const char *message), set separately for errors and info.
    jack = _load("libjack.so.0", ctypes.util.find_library("jack") or "libjack.so")
    if jack is not None:
        try:
            jack_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p)(lambda *args: None)
            jack.jack_set_error_function(jack_handler)
            jack.jack_set_info_function(jack_handler)
            _keep_alive.append(jack_handler)
            silenced = True
        except Exception:
            logger.debug("Could not silence JACK messages", exc_info=True)

    _installed = silenced
    return silenced
