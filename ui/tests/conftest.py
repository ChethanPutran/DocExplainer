import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed

import gc
import time

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pump(qapp):
    """Process Qt events until ``cond()`` is true (callbacks are delivered through the event loop)."""
    def _pump(cond, timeout=8.0):
        end = time.time() + timeout
        while time.time() < end:
            QCoreApplication.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False
    return _pump


@pytest.fixture(autouse=True)
def _collect_garbage_on_the_main_thread():
    """Qt objects caught in reference cycles must be destroyed on the GUI thread. If the garbage
    collector instead runs inside a worker thread (any allocation can trigger it), Qt aborts."""
    yield
    gc.collect()
