"""The viewers must actually SHOW the document (a duplicated _setup_ui once left the text in an invisible widget)."""
import pymupdf
import pytest

from doc_explainer_ui.gui.factories.viewer_factory import ViewerFactory
from doc_explainer_ui.gui.widgets import HTMLViewer, PDFViewer, TextViewer


def shown(viewer):
    """The viewer's widget tree contains exactly one content widget and it is the one that holds the text."""
    viewer.resize(600, 400)
    viewer.show()
    return viewer


def test_text_viewer_shows_the_text(qapp, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Hello photosynthesis\nSecond line")
    v = shown(TextViewer())
    assert v.load(str(f))
    assert v.layout().count() == 1 and v.layout().itemAt(0).widget() is v.text_edit      # the one on screen
    assert "Hello photosynthesis" in v.text_edit.toPlainText() and v.text_edit.isVisible()


def test_html_viewer_shows_the_page(qapp, tmp_path):
    f = tmp_path / "a.html"
    f.write_text("<html><body><h1>Title here</h1><p>Body text</p></body></html>")
    v = shown(HTMLViewer())
    assert v.load(str(f))
    assert v.layout().count() == 1 and v.layout().itemAt(0).widget() is v.browser
    assert "Title here" in v.browser.toPlainText() and v.browser.isVisible()


def test_pdf_viewer_renders_every_page_and_can_jump(qapp, tmp_path):
    f = tmp_path / "a.pdf"
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page number {i + 1}")
    doc.save(str(f))
    v = shown(PDFViewer())
    assert v.load(str(f))
    assert v.pages_layout.count() == 3 and v.layout().itemAt(0).widget() is v.scroll_area
    assert v.go_to_page(2) and v.current_page == 2 and not v.go_to_page(9)


def test_markdown_files_get_a_viewer(qapp, tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Heading\n\ntext")
    v = shown(ViewerFactory.create_viewer(str(f)))
    assert v.load(str(f)) and "Heading" in v.text_edit.toPlainText()


def test_selecting_text_emits_the_viewer_signal_without_error(qapp, tmp_path, pump):
    f = tmp_path / "a.txt"
    f.write_text("select me please")
    v = shown(TextViewer())
    v.load(str(f))
    got = []
    v.text_selected.connect(lambda *a: got.append(a))
    cursor = v.text_edit.textCursor()
    cursor.select(cursor.SelectionType.Document)
    v.text_edit.setTextCursor(cursor)
    assert pump(lambda: got) and got[0][0] == "text_selected" and got[0][2] == "select me please"


# ----------------------------------------------------------------------------- right-click menu
from PySide6.QtCore import QPoint  # noqa: E402

from doc_explainer_ui.gui.models.signals import UISignals  # noqa: E402
from doc_explainer_ui.gui.widgets.base import base_viewer  # noqa: E402


def texts(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def enabled(menu):
    return {a.text(): a.isEnabled() for a in menu.actions() if not a.isSeparator()}


def text_viewer(tmp_path, signals=None, doc_id=""):
    f = tmp_path / "a.txt"
    f.write_text("Calvin cycle uses ATP to fix carbon")
    v = shown(TextViewer(signals=signals or UISignals()))
    v.load(str(f))
    v.doc_id = doc_id
    return v


def select_all(v):
    c = v.text_edit.textCursor()
    c.select(c.SelectionType.Document)
    v.text_edit.setTextCursor(c)


def test_right_click_is_wired_to_every_viewer(qapp):
    from PySide6.QtCore import Qt
    for v, inner in ((TextViewer(), "text_edit"), (HTMLViewer(), "browser"), (PDFViewer(), "scroll_area")):
        assert getattr(v, inner).contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_before_the_document_is_ready_only_the_ai_actions_are_disabled(qapp, tmp_path):
    sig, got = UISignals(), []
    sig.explain_requested.connect(lambda *a: got.append(a))
    v = text_viewer(tmp_path, sig)
    select_all(v)
    menu = v._build_menu(QPoint(5, 5))
    state = enabled(menu)
    assert state["Copy"] and state["Select all"]                                       # reading tools work
    assert not state["Explain this selection"] and not state["Summarize this selection"]
    assert not state["Ask a question about this selection..."]                          # AI is greyed, not hidden
    hint = texts(menu)[-1]
    assert "being processed" in hint and "available when ready" in hint and not state[hint]
    next(a for a in menu.actions() if a.text().startswith("Explain")).trigger()        # a disabled item does nothing
    assert got == []
    v.ai_note = "Document is being processed: 42%"
    assert "42%" in texts(v._build_menu(QPoint(5, 5)))[-1]
    v.ai_note = "Document could not be processed, so the AI tutor is not available: cannot parse"
    assert "cannot parse" in texts(v._build_menu(QPoint(5, 5)))[-1]


def test_select_all_from_the_menu_works_before_the_document_is_ready(qapp, tmp_path):
    v = text_viewer(tmp_path)
    next(a for a in v._build_menu(QPoint(5, 5)).actions() if a.text() == "Select all").trigger()
    assert v.text_edit.textCursor().selectedText() == "Calvin cycle uses ATP to fix carbon"


def test_menu_when_ready_offers_the_ai_actions_on_the_selection(qapp, tmp_path):
    sig = UISignals()
    got = {"explain": [], "summarize": []}
    sig.explain_requested.connect(lambda *a: got["explain"].append(a))
    sig.summarize_requested.connect(lambda *a: got["summarize"].append(a))
    v = text_viewer(tmp_path, sig, doc_id="d1")
    menu = v._build_menu(QPoint(5, 5))                                   # nothing selected
    assert texts(menu) == ["Copy", "Select all", "Explain this selection", "Summarize this selection",
                           "Ask a question about this selection...", "Save this selection to notes", "Bookmark this place",
                           "Highlight selection", "Clear highlights"]
    assert not enabled(menu)["Copy"] and not enabled(menu)["Explain this selection"] and enabled(menu)["Ask a question about this selection..."]
    select_all(v)
    menu = v._build_menu(QPoint(5, 5))
    assert all(enabled(menu).values())
    next(a for a in menu.actions() if a.text().startswith("Explain")).trigger()
    next(a for a in menu.actions() if a.text().startswith("Summarize")).trigger()
    assert got["explain"][0][:2] == ("d1", "Calvin cycle uses ATP to fix carbon")
    assert got["summarize"][0][:2] == ("d1", "Calvin cycle uses ATP to fix carbon")


def test_ask_sends_the_typed_question_with_the_selected_passage(qapp, tmp_path, monkeypatch):
    sig, got = UISignals(), []
    sig.ask_requested.connect(lambda *a: got.append(a))
    v = text_viewer(tmp_path, sig, doc_id="d1")
    select_all(v)
    ask = next(a for a in v._build_menu(QPoint(5, 5)).actions() if a.text().startswith("Ask"))
    monkeypatch.setattr(base_viewer.QInputDialog, "getText", staticmethod(lambda *a: ("Why ATP?", True)))
    ask.trigger()
    assert got[0][0] == "d1" and got[0][1].startswith("Why ATP?") and "Calvin cycle uses ATP" in got[0][1]
    got.clear()
    monkeypatch.setattr(base_viewer.QInputDialog, "getText", staticmethod(lambda *a: ("", True)))     # empty = nothing sent
    ask.trigger()
    monkeypatch.setattr(base_viewer.QInputDialog, "getText", staticmethod(lambda *a: ("x", False)))   # cancelled
    ask.trigger()
    assert got == []


def test_a_question_without_a_selection_is_sent_as_is(qapp, tmp_path, monkeypatch):
    sig, got = UISignals(), []
    sig.ask_requested.connect(lambda *a: got.append(a))
    v = text_viewer(tmp_path, sig, doc_id="d1")
    ask = next(a for a in v._build_menu(QPoint(5, 5)).actions() if a.text().startswith("Ask"))
    monkeypatch.setattr(base_viewer.QInputDialog, "getText", staticmethod(lambda *a: ("What is this about?", True)))
    ask.trigger()
    assert got[0][1] == "What is this about?"


def test_pdf_menu_acts_on_the_page_that_was_clicked(qapp, tmp_path):
    f = tmp_path / "a.pdf"
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Content of page {i + 1}")
    doc.save(str(f))
    sig, got = UISignals(), []
    sig.explain_requested.connect(lambda *a: got.append(a))
    v = shown(PDFViewer(signals=sig))
    v.load(str(f))
    v.doc_id = "d1"
    qapp.processEvents()
    second = v.pages_layout.itemAt(1).widget()
    point = v.scroll_area.viewport().mapFrom(v.pages_container, second.geometry().center())
    menu = v._build_menu(point)
    assert texts(menu) == ["Explain this page", "Summarize this page", "Ask a question about this page...",
                           "Save this page to notes", "Bookmark this place"]   # no Copy: a PDF page is an image
    menu.actions()[0].trigger()
    assert got[0][0] == "d1" and "Content of page 2" in got[0][1] and got[0][2] == 2


def test_long_pages_are_trimmed_before_they_are_sent(qapp, tmp_path):
    sig, got = UISignals(), []
    sig.explain_requested.connect(lambda *a: got.append(a))
    f = tmp_path / "long.txt"
    f.write_text("word " * 5000)
    v = shown(TextViewer(signals=sig))
    v.load(str(f))
    v.doc_id = "d1"
    select_all(v)
    next(a for a in v._build_menu(QPoint(1, 1)).actions() if a.text().startswith("Explain")).trigger()
    assert len(got[0][1]) == v.MAX_CONTEXT_CHARS


def test_menu_saves_the_selection_to_notes_bookmarks_and_highlights(qapp, tmp_path):
    sig, notes = UISignals(), []
    sig.note_requested.connect(lambda d, t: notes.append((d, t)))
    v = text_viewer(tmp_path, sig, doc_id="d1")
    select_all(v)
    menu = v._build_menu(QPoint(5, 5))
    next(a for a in menu.actions() if a.text().startswith("Save this")).trigger()
    next(a for a in menu.actions() if a.text().startswith("Bookmark")).trigger()
    assert notes[0][0] == "d1" and "page 1" in notes[0][1] and notes[1][1].startswith("\U0001F516 Bookmark")
    next(a for a in menu.actions() if a.text().startswith("Highlight")).trigger()
    assert len(v.text_edit.extraSelections()) == 1
    v.clear_highlights()
    assert v.text_edit.extraSelections() == []


def test_the_default_theme_is_applied_at_start_up(qapp):
    from doc_explainer_ui.gui.managers.theme_manager import ThemeManager
    qapp.setStyleSheet("")
    tm = ThemeManager(qapp)
    assert tm.set_theme("light") and qapp.styleSheet()          # naming the initial theme must still style the app
