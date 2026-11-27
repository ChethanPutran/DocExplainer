"""Progress dashboard, chat export, drag and drop."""
import time

from PySide6.QtCore import QMimeData, QUrl

from doc_explainer_ui.gui.managers.window_manager import WindowManager
from doc_explainer_ui.gui.windows.progress_window import ProgressWindow, mastery_series

HISTORY = [{"at": 100.0, "concept_id": "a", "after": 0.4}, {"at": 200.0, "concept_id": "b", "after": 0.8},
           {"at": 300.0, "concept_id": "a", "after": 0.6}]


def test_mastery_series_averages_the_latest_belief_of_every_practised_concept():
    assert mastery_series(HISTORY) == [(100.0, 0.4), (200.0, 0.6000000000000001), (300.0, 0.7)] or \
        [round(v, 3) for _, v in mastery_series(HISTORY)] == [0.4, 0.6, 0.7]
    assert mastery_series([]) == []


def test_dashboard_shows_level_concepts_and_what_is_due(qapp):
    w = ProgressWindow()
    learner = {"level": "intermediate", "concepts": [{"concept_id": "a", "name": "ATP", "p_known": 0.3, "evidence": 2},
                                                     {"concept_id": "b", "name": "Calvin cycle", "p_known": 0.9, "evidence": 1},
                                                     {"concept_id": "c", "name": "Unseen", "p_known": 0.5, "evidence": 0}]}
    review = [{"concept_id": "a", "name": "ATP", "p_known": 0.3, "retention": 0.2, "last_seen": time.time(), "due": True}]
    w.set_data(learner, HISTORY, review)
    assert "Intermediate" in w.level.text() and "2 concepts practised" in w.summary.text() and "1 due" in w.summary.text()
    assert w.concepts.rowCount() == 3 and w.concepts.item(0, 0).text() == "ATP"          # practised first, weakest first
    assert w.review.rowCount() == 1 and w.practise_button.isEnabled()
    assert len(w.chart.series) == 3
    w.chart.grab()                                                                          # paints without error
    w.set_data(None, [], [])
    assert "No knowledge tracking" in w.level.text() and not w.practise_button.isEnabled() and w.concepts.rowCount() == 0


def test_a_conversation_is_saved_as_markdown_or_pdf(tmp_path, qapp):
    md = "# Chat\n\n**You**\n\nWhat is ATP?\n"
    WindowManager.write_export(str(tmp_path / "c.md"), md)
    assert (tmp_path / "c.md").read_text() == md
    WindowManager.write_export(str(tmp_path / "c.pdf"), md)
    assert (tmp_path / "c.pdf").read_bytes().startswith(b"%PDF")


def test_dropping_a_supported_file_opens_it(qapp, tmp_path):
    from types import SimpleNamespace
    from doc_explainer_ui.gui.windows import main_window as mw
    opened = []
    fake = SimpleNamespace(SUPPORTED=mw.MainWindow.SUPPORTED, open_document=opened.append)

    class Event:
        def __init__(self, urls):
            self._m = QMimeData()
            self._m.setUrls([QUrl.fromLocalFile(u) for u in urls])
            self.accepted = False

        def mimeData(self):
            return self._m

        def acceptProposedAction(self):
            self.accepted = True
    ev = Event([str(tmp_path / "a.pdf"), str(tmp_path / "b.exe"), str(tmp_path / "c.DOCX")])
    mw.MainWindow.dropEvent(fake, ev)
    assert [p.rsplit("/", 1)[-1] for p in opened] == ["a.pdf", "c.DOCX"] and ev.accepted
    ev2 = Event([str(tmp_path / "x.exe")])
    mw.MainWindow.dragEnterEvent(fake, ev2)
    assert not ev2.accepted
