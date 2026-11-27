"""Background: slow work runs off the GUI thread, results come back on it."""
import threading
import time

from doc_explainer_ui.gui.utils.background import Background, Cancelled


def test_result_is_delivered_on_the_gui_thread_and_work_runs_elsewhere(qapp, pump):
    bg, seen = Background(), {}

    def work(emit, cancel):
        seen["work"] = threading.current_thread()
        time.sleep(0.05)
        return 42

    bg.run(work, on_done=lambda r: seen.update(done=threading.current_thread(), result=r))
    assert "result" not in seen                      # run() returned without waiting
    assert pump(lambda: "result" in seen)
    assert seen["result"] == 42
    assert seen["done"] is threading.main_thread() and seen["work"] is not threading.main_thread()
    assert bg.active == 0


def test_errors_and_progress_are_delivered_on_the_gui_thread(qapp, pump):
    bg, got = Background(), {"progress": []}

    def work(emit, cancel):
        emit("a"); emit("b")
        raise ValueError("boom")

    bg.run(work, on_progress=lambda p: got["progress"].append((p, threading.current_thread())),
           on_error=lambda e: got.update(error=e, thread=threading.current_thread()))
    assert pump(lambda: "error" in got)
    assert isinstance(got["error"], ValueError) and got["thread"] is threading.main_thread()
    pump(lambda: len(got["progress"]) == 2, 2)
    assert [p for p, _ in got["progress"]] == ["a", "b"]
    assert all(t is threading.main_thread() for _, t in got["progress"])


def test_a_failing_callback_does_not_break_other_tasks(qapp, pump):
    bg, ok = Background(), []
    bg.run(lambda e, c: 1, on_done=lambda r: 1 / 0)             # callback raises: logged, swallowed
    bg.run(lambda e, c: 2, on_done=ok.append)
    assert pump(lambda: ok == [2])


def test_shutdown_cancels_cooperating_work_without_reporting_an_error(qapp, pump):
    bg, flags = Background(), {"stopped": False, "errors": []}

    def work(emit, cancel):
        while not cancel.is_set():
            time.sleep(0.01)
        flags["stopped"] = True
        raise Cancelled()

    bg.run(work, on_error=flags["errors"].append)
    time.sleep(0.05)
    bg.shutdown()
    assert pump(lambda: flags["stopped"] and bg.active == 0)
    assert flags["errors"] == []                       # cancellation is not an error to show the user


def test_threads_are_daemons_so_exit_never_waits_on_them(qapp):
    bg, release = Background(), threading.Event()
    bg.run(lambda e, c: release.wait(5))
    t = [t for t in threading.enumerate() if t.name == "doc-explainer-bg"][0]
    assert t.daemon
    release.set()
