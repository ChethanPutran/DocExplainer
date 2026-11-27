"""MainWindow.open_document: the document shows at once; the tab tracks the background processing."""
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QTabWidget, QWidget

from doc_explainer_ui.config import UIConfig
from doc_explainer_ui.gui.models.signals import UISignals
from doc_explainer_ui.gui.widgets import BaseDocumentViewer, Sidebar
from doc_explainer_ui.gui.windows import main_window as mw


class StubViewer(BaseDocumentViewer):
    def __init__(self, loads=True):
        self._loads = loads
        super().__init__()

    def load(self, path):
        return self._loads

    def clear(self):
        return True


class FakeWM:
    def __init__(self):
        self.calls, self.progress, self.shown = [], [], []

    def register_document_async(self, path, on_done=None, on_error=None, on_progress=None, on_cancelled=None, build_graph=True):
        self.calls.append((path, on_done, on_error))
        self.build_graph = build_graph
        self.progress.append(on_progress)
        self.cancelled = on_cancelled

    def show_document(self, doc_id):
        self.shown.append(doc_id)


@pytest.fixture
def env(qapp, monkeypatch):
    viewers, warnings = [], []
    monkeypatch.setattr(mw.ViewerFactory, "create_viewer", staticmethod(lambda p, signals=None: viewers.append(StubViewer()) or viewers[-1]))
    monkeypatch.setattr(mw.QMessageBox, "warning", staticmethod(lambda *a: warnings.append(a[1:])))
    monkeypatch.setattr(mw.QMessageBox, "critical", staticmethod(lambda *a: warnings.append(a[1:])))
    opened = []
    signals = UISignals()
    signals.document_opened.connect(lambda d, p: opened.append((d, p)))
    window = SimpleNamespace(tabs=QTabWidget(), signals=signals, window_manager=FakeWM(), sidebar=Sidebar(),
                             config=UIConfig())
    window._wants_graph = lambda path: mw.MainWindow._wants_graph(window, path)
    window._refresh_ai = lambda: mw.MainWindow._refresh_ai(window)       # the real gating logic, without the full window
    return SimpleNamespace(w=window, viewers=viewers, warnings=warnings, opened=opened)


def open_(env, path):
    mw.MainWindow.open_document(env.w, path)   # (no self-referencing lambda: that made a Qt cycle)


def test_tab_appears_immediately_marked_as_processing_then_becomes_ready(env):
    open_(env, "/x/paper.pdf")
    assert env.w.tabs.count() == 1 and env.w.tabs.tabText(0).endswith("paper.pdf") and "⏳" in env.w.tabs.tabText(0)
    assert env.viewers[0].doc_id == ""                       # not usable for questions yet
    _, on_done, _ = env.w.window_manager.calls[0]
    on_done("doc-9")
    assert env.w.tabs.tabText(0) == "paper.pdf" and env.viewers[0].doc_id == "doc-9"
    assert env.w.window_manager.shown == ["doc-9"]           # the Learn and Resources tabs are filled for it
    assert env.opened == [("doc-9", "/x/paper.pdf")]


def test_tab_title_shows_processing_progress(env):
    open_(env, "/x/paper.pdf")
    env.w.window_manager.progress[0](SimpleNamespace(state="running", progress=0.42, stage="reading section 2 of 5"))
    assert "42%" in env.w.tabs.tabText(0)


def test_ai_features_are_hidden_until_the_document_is_ready_on_the_server(env):
    open_(env, "/x/paper.pdf")
    w = env.w
    w._refresh_ai()
    assert not w.sidebar.available and not w.sidebar.chat_panel.input.isEnabled()          # nothing yet
    w.window_manager.progress[0](SimpleNamespace(state="running", progress=0.42, stage="reading section 2 of 5"))
    text = w.sidebar.unavailable_label.text()
    assert not w.sidebar.available and "42%" in text and "section 2 of 5" in text and "read the document now" in text
    assert not w.sidebar.chat_panel.input.isEnabled()
    _, on_done, _ = w.window_manager.calls[0]
    on_done("doc-9")
    assert w.sidebar.available and w.sidebar.chat_panel.input.isEnabled()                  # now offered


def test_a_failed_document_never_gets_ai_features_and_says_why(env):
    open_(env, "/x/paper.pdf")
    _, _, on_error = env.w.window_manager.calls[0]
    on_error("cannot parse")
    assert not env.w.sidebar.available and "cannot parse" in env.w.sidebar.unavailable_label.text()
    assert not env.w.sidebar.chat_panel.input.isEnabled()


def test_with_no_document_open_there_is_nothing_to_ask(env):
    env.w._refresh_ai()
    assert not env.w.sidebar.available and "Open a document" in env.w.sidebar.unavailable_label.text()


def test_failure_marks_the_tab_and_explains_why(env):
    open_(env, "/x/paper.pdf")
    _, _, on_error = env.w.window_manager.calls[0]
    on_error("cannot parse")
    assert "⚠" in env.w.tabs.tabText(0) and env.viewers[0].doc_id == ""
    assert env.warnings and "cannot parse" in str(env.warnings[0])
    assert env.opened == []


def test_closing_the_tab_before_processing_finishes_is_harmless(env):
    open_(env, "/x/paper.pdf")
    _, on_done, on_error = env.w.window_manager.calls[0]
    env.w.tabs.removeTab(0)
    on_done("doc-9")                                          # no tab any more: must not raise
    on_error("late failure")
    assert env.opened == [] and env.warnings == []


def test_a_file_the_viewer_cannot_load_opens_no_tab_and_registers_nothing(env, monkeypatch):
    monkeypatch.setattr(mw.ViewerFactory, "create_viewer", staticmethod(lambda p, signals=None: StubViewer(loads=False)))
    open_(env, "/x/broken.pdf")
    assert env.w.tabs.count() == 0 and env.w.window_manager.calls == [] and env.warnings


def test_a_paused_document_shows_a_pause_marker_and_says_how_to_continue(env):
    open_(env, "/x/paper.pdf")
    env.w.window_manager.progress[0](SimpleNamespace(state="paused", progress=0.4, stage="paused"))
    assert "\u23f8" in env.w.tabs.tabText(0)
    assert "paused" in env.w.sidebar.unavailable_label.text() and "Job status" in env.w.sidebar.unavailable_label.text()
    assert not env.w.sidebar.available


def test_a_cancelled_document_is_marked_without_an_error_dialog(env):
    open_(env, "/x/paper.pdf")
    env.w.window_manager.cancelled()
    assert "\u26d4" in env.w.tabs.tabText(0) and env.warnings == []
    assert "cancelled" in env.w.sidebar.unavailable_label.text() and not env.w.sidebar.available


def test_knowledge_graph_choice_follows_the_setting_and_asks_once_per_new_file(env, monkeypatch):
    asked = []
    monkeypatch.setattr(mw.QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[1]) or mw.QMessageBox.StandardButton.No))
    wm = env.w.window_manager
    open_(env, "/x/a.pdf")
    assert wm.build_graph is True and not asked                       # default: always
    env.w.config.knowledge.build_graph = "never"
    open_(env, "/x/b.pdf")
    assert wm.build_graph is False and not asked
    env.w.config.knowledge.build_graph = "ask"
    open_(env, "/x/c.pdf")
    assert wm.build_graph is False and len(asked) == 1                # the user said no
    wm.is_known_file = lambda p: True
    open_(env, "/x/c.pdf")
    assert len(asked) == 1                                            # a file the server already has: no question
