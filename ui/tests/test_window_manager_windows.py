"""The Library and Progress windows opened through the real WindowManager, against a mock server."""
from types import SimpleNamespace

import httpx

from doc_explainer_ui.gui.utils.recent_files import RecentFiles
from test_service_actions import make_wm

DOC = {"doc_id": "d1", "title": "Book", "filename": "book.pdf", "size_bytes": 10, "uploaded_at": "2026-01-01T00:00:00",
       "knowledge": "ready", "sections": 2, "concepts": 3}


class Library:
    def __init__(self):
        self.docs, self.requests = [dict(DOC)], []

    def __call__(self, req):
        self.requests.append((req.method, req.url.path))
        p = req.url.path
        if req.method == "GET" and p == "/v1/documents":
            return httpx.Response(200, json={"documents": self.docs})
        if req.method == "GET" and p == "/v1/documents/d1":
            return httpx.Response(200, json=self.docs[0])
        if req.method == "DELETE" and p == "/v1/documents/d1/knowledge":
            self.docs[0].update(knowledge="none", concepts=0)
            return httpx.Response(204)
        if req.method == "DELETE" and p == "/v1/documents/d1":
            self.docs.clear()
            return httpx.Response(204)
        if p == "/v1/documents/d1/learner":
            return httpx.Response(200, json={"level": "beginner", "prior": 0.3, "position": 1,
                                             "concepts": [{"concept_id": "c1", "name": "ATP", "p_known": 0.5, "evidence": 1}]})
        if p == "/v1/documents/d1/history":
            return httpx.Response(200, json={"points": [{"at": 1.0, "kind": "quiz_answer", "concept_id": "c1", "name": "ATP", "before": 0.3, "after": 0.5}]})
        if p == "/v1/documents/d1/review":
            return httpx.Response(200, json={"items": [{"concept_id": "c1", "name": "ATP", "p_known": 0.5, "retention": 0.2, "last_seen": 1.0, "due": True}]})
        return httpx.Response(404, json={"error": {"code": "document_not_found", "message": "no"}})


def setup(tmp_path):
    server = Library()
    wm, path = make_wm(server, tmp_path)
    wm.recents = RecentFiles(tmp_path / "recent.json")
    wm.library_window = wm.progress_window = None
    wm.main_window = SimpleNamespace(focus_document=lambda d: False, close_document=lambda d: None, statusBar=lambda: SimpleNamespace(showMessage=lambda *a: None))
    wm._status = lambda *a, **k: None
    wm._notify_error = lambda *a, **k: None
    return wm, server


def test_the_library_opens_lists_documents_and_removes_a_graph(qapp, tmp_path, pump):
    wm, server = setup(tmp_path)
    wm.show_library()                                          # used to crash: a signal it connects did not exist
    assert pump(lambda: wm.library_window.table.rowCount() == 1)
    wm.library_window.remove_graph_requested.emit("d1")
    assert pump(lambda: ("DELETE", "/v1/documents/d1/knowledge") in server.requests)
    assert pump(lambda: wm.library_window.table.item(0, 6).text() == "Not built")


def test_deleting_from_the_library_calls_the_server_and_refreshes(qapp, tmp_path, pump):
    wm, server = setup(tmp_path)
    wm.show_library()
    assert pump(lambda: wm.library_window.table.rowCount() == 1)
    wm.library_window.delete_requested.emit("d1")
    assert pump(lambda: wm.library_window.table.rowCount() == 0)


def test_the_progress_window_loads_learner_history_and_review(qapp, tmp_path, pump):
    wm, server = setup(tmp_path)
    wm._registered.add("d1")
    wm.show_progress("d1")
    assert pump(lambda: wm.progress_window.concepts.rowCount() == 1)
    assert "Beginner" in wm.progress_window.level.text() and wm.progress_window.review.rowCount() == 1
