"""Run slow service calls off the Qt thread and deliver results back on it.

Design notes (these are the bugs this shape avoids):

* Daemon threads, not QThreadPool: a call still waiting on the network must never keep the
  application from exiting.
* Workers only ever touch ONE long-lived QObject (the hub), created on the GUI thread and never
  destroyed while the program runs. Per-task state is a plain Python object. If each task owned a
  QObject, the worker thread could hold the last reference and destroy it, and Qt aborts when a
  QObject is destroyed on a thread that does not own it.
* The hub's slot runs on the GUI thread, so the callbacks you pass always run there and may touch
  widgets.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Set

from PySide6.QtCore import QObject, Signal, Slot

logger = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised by work functions that notice ``cancel`` is set (the application is closing)."""


class _Task:
    """Plain Python state for one background call (safe to drop from any thread)."""

    def __init__(self, on_done, on_error, on_progress, finished: Callable[["_Task"], None]):
        self.on_done, self.on_error, self.on_progress, self.finished = on_done, on_error, on_progress, finished

    def deliver(self, kind: str, payload: Any) -> None:  # GUI thread
        if kind != "progress":
            self.finished(self)
        if kind == "failed" and isinstance(payload, Cancelled):
            return
        callback = {"done": self.on_done, "failed": self.on_error, "progress": self.on_progress}[kind]
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:  # a bug in a callback must not take the UI down
            logger.exception("Error in a background-task callback")


class _Hub(QObject):
    event = Signal(object)  # (task, kind, payload); queued onto the GUI thread

    def __init__(self) -> None:
        super().__init__()
        self.event.connect(self._dispatch)

    @Slot(object)
    def _dispatch(self, ev) -> None:
        task, kind, payload = ev
        task.deliver(kind, payload)


_hub: Optional[_Hub] = None


def _get_hub() -> _Hub:
    """Created lazily, on the thread that first needs it (the GUI thread), and kept for good."""
    global _hub
    if _hub is None:
        _hub = _Hub()
    return _hub


class Background:
    """Owns the running tasks. Create and call it from the GUI thread."""

    def __init__(self) -> None:
        self._hub = _get_hub()
        self._live: Set[_Task] = set()
        self._cancel = threading.Event()

    def run(self, work: Callable[[Callable[[Any], None], threading.Event], Any], *,
            on_done: Optional[Callable[[Any], None]] = None,
            on_error: Optional[Callable[[BaseException], None]] = None,
            on_progress: Optional[Callable[[Any], None]] = None) -> None:
        """Run ``work(emit, cancel)`` on a worker thread.

        ``emit(update)`` sends a progress update to ``on_progress``; long loops should check
        ``cancel.is_set()`` and raise ``Cancelled``. ``on_done(result)`` / ``on_error(exc)`` run on
        the GUI thread (a ``Cancelled`` is swallowed: closing the app is not an error to show).
        """
        task = _Task(on_done, on_error, on_progress, self._live.discard)
        self._live.add(task)
        hub = self._hub

        def send(kind: str, payload: Any) -> None:
            try:
                hub.event.emit((task, kind, payload))
            except RuntimeError:  # the hub's C++ side is gone: the interpreter is shutting down
                pass

        def target() -> None:
            try:
                result = work(lambda update: send("progress", update), self._cancel)
            except BaseException as e:  # noqa: BLE001 - reported to the GUI thread
                send("failed", e)
            else:
                send("done", result)

        threading.Thread(target=target, name="doc-explainer-bg", daemon=True).start()

    @property
    def active(self) -> int:
        return len(self._live)

    def shutdown(self) -> None:
        """Ask long-running work to stop. Threads are daemons, so exit never blocks on them."""
        self._cancel.set()
