"""Recent files and the library window."""
import json

import pytest

from doc_explainer_ui.gui.utils.recent_files import RecentFiles
from doc_explainer_ui.gui.windows.library_window import LibraryWindow, human_size


def test_recent_files_newest_first_without_duplicates_and_limited(tmp_path):
    files = []
    for n in "abcd":
        f = tmp_path / f"{n}.txt"
        f.write_text(n)
        files.append(str(f))
    r = RecentFiles(tmp_path / "recent.json", limit=3)
    for f in files:
        r.add(f)
    r.add(files[2])                                          # opened again: moves to the top, no copy
    assert r.paths() == [files[2], files[3], files[1]]
    assert RecentFiles(tmp_path / "recent.json", limit=3).paths() == r.paths()      # persisted
    (tmp_path / "d.txt").unlink()
    assert files[3] not in r.paths()                         # files that no longer exist are not offered


def test_library_links_survive_and_forgetting_a_document_drops_its_recent_entry(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_text("x")
    r = RecentFiles(tmp_path / "recent.json")
    r.add(str(f))
    r.link("doc1", str(f))
    again = RecentFiles(tmp_path / "recent.json")
    assert again.path_for("doc1") == str(f)
    again.forget_document("doc1")
    assert again.path_for("doc1") is None and again.paths() == []


def test_old_recent_file_format_still_loads(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    (tmp_path / "recent.json").write_text(json.dumps({"documents": [str(f)]}))
    assert RecentFiles(tmp_path / "recent.json").paths() == [str(f)]


DOCS = [{"doc_id": "1", "title": "Old", "filename": "old.pdf", "size_bytes": 2048, "uploaded_at": "2026-01-01T10:00:00", "knowledge": "ready", "sections": 3, "concepts": 9},
        {"doc_id": "2", "title": "Newer", "filename": "new.txt", "size_bytes": 10, "uploaded_at": "2026-02-01T10:00:00", "knowledge": "partial"}]


def test_library_lists_newest_first_filters_and_emits_open(qapp):
    w, opened = LibraryWindow(), []
    w.open_requested.connect(opened.append)
    w.set_documents(DOCS)
    assert [w.table.item(r, 0).text() for r in range(2)] == ["Newer", "Old"]
    w.search.setText("old")
    assert w.table.rowCount() == 1 and w.table.item(0, 0).text() == "Old"
    w.open_button.click()
    assert opened == ["1"]


def test_library_delete_asks_first(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w, deleted = LibraryWindow(), []
    w.delete_requested.connect(deleted.append)
    w.set_documents(DOCS)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w.delete_button.click()
    assert deleted == []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    w.delete_button.click()
    assert deleted == ["2"]


def test_library_empty_state_disables_buttons(qapp):
    w = LibraryWindow()
    w.set_documents([])
    assert not w.open_button.isEnabled() and not w.delete_button.isEnabled() and "No documents" in w.status.text()
    assert human_size(2048) == "2.0 KB"


def test_every_library_signal_the_window_manager_connects_exists(qapp):
    """show_library connects these; a missing one crashed the app when the Library was opened."""
    w = LibraryWindow()
    for name in ("refresh_requested", "open_requested", "delete_requested", "build_graph_requested", "remove_graph_requested"):
        assert hasattr(w, name), name


def test_removing_a_graph_asks_first_and_only_for_documents_that_have_one(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w, removed = LibraryWindow(), []
    w.remove_graph_requested.connect(removed.append)
    w.set_documents(DOCS)                                              # row 0 = "Newer" (partial), row 1 = "Old" (ready)
    assert w.drop_graph_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w.drop_graph_button.click()
    assert removed == []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    w.drop_graph_button.click()
    assert removed == ["2"]
    w.set_documents([{**DOCS[0], "knowledge": "none"}])
    assert not w.drop_graph_button.isEnabled() and w.graph_button.isEnabled()
