"""ListeningThread / VoiceInput, with a scripted microphone and a fake model whose timing we control.

The behaviours under test are the ones that used to lose the user's first words:
  - the microphone is live while the model is still loading, and nothing said is lost;
  - listening continues while a clip is being transcribed;
  - stopping neither blocks the UI nor throws away phrases already spoken;
  - a second session reuses the model that is already in memory.
"""
import threading
import time
import types
import wave

import pytest
import speech_recognition as sr
from PySide6.QtCore import QCoreApplication, QObject, Signal

from doc_explainer_ui.config import VoiceConfig
from doc_explainer_ui.gui.widgets.voice import voice_input as vi
from doc_explainer_ui.gui.widgets.voice.speech_model import WhisperModelCache


def phrase(text: str):
    data = text.encode()
    data += b"\x00" * (len(data) % 2)                      # 16-bit samples need an even length
    return sr.AudioData(data, 16000, 2)


class FakeMic:
    SAMPLE_RATE, CHUNK = 16000, 1024

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeRecognizer:
    """`listen` hands out the scripted phrases, then times out (so the loop keeps polling)."""
    script = []
    listened = 0

    def __init__(self):
        self.energy_threshold = 300

    def adjust_for_ambient_noise(self, source, duration=0.5):
        pass

    def listen(self, source, timeout=None, phrase_time_limit=None):
        FakeRecognizer.listened += 1
        if FakeRecognizer.script:
            return FakeRecognizer.script.pop(0)
        time.sleep(0.01)
        raise sr.WaitTimeoutError()


class FakeModel:
    """Understands the 'audio' written by `phrase`: transcribes it back to its text."""

    def __init__(self, delay=0.0, gate=None):
        self.delay, self.gate, self.calls = delay, gate, 0

    def transcribe(self, path, fp16=False):
        self.calls += 1
        if self.gate is not None:
            assert self.gate.wait(10), "test forgot to release the transcription gate"
        time.sleep(self.delay)
        with wave.open(path) as w:
            raw = w.readframes(w.getnframes())
        return {"text": raw.rstrip(b"\x00").decode()}


@pytest.fixture(autouse=True)
def fake_audio(qapp, monkeypatch):
    FakeRecognizer.script, FakeRecognizer.listened = [], 0
    monkeypatch.setattr(vi.sr, "Microphone", FakeMic)
    monkeypatch.setattr(vi.sr, "Recognizer", FakeRecognizer)


def make_cache(model=None, load_gate=None, fail=False):
    loads = []

    def loader(name):
        loads.append(name)
        if load_gate is not None:
            assert load_gate.wait(10), "test forgot to release the model-load gate"
        if fail:
            raise RuntimeError("no such model")
        return model or FakeModel()

    return WhisperModelCache(loader=loader, idle_unload_s=600), loads


class Run:
    """Run a ListeningThread's capture loop on a plain thread and record what it emits."""

    def __init__(self, cache, model_name="base"):
        self.thread = vi.ListeningThread(model_name, cache=cache)
        self.texts, self.errors, self.events = [], [], []
        t = self.thread
        t.text_recognized.connect(self.texts.append)
        t.error_occurred.connect(self.errors.append)
        t.listening_started.connect(lambda: self.events.append("mic"))
        t.model_state.connect(lambda s: self.events.append(f"model:{s}"))
        self.runner = threading.Thread(target=t.run, daemon=True)
        self.runner.start()

    def wait(self, cond, timeout=6.0):
        end = time.time() + timeout
        while time.time() < end:
            QCoreApplication.processEvents()   # signals from other threads are queued onto this one
            if cond():
                return True
            time.sleep(0.01)
        return False

    def stop(self):
        self.thread.stop()
        self.runner.join(5)
        assert not self.runner.is_alive()
        QCoreApplication.processEvents()


def test_microphone_is_live_while_the_model_loads_and_no_words_are_lost():
    gate = threading.Event()
    cache, loads = make_cache(load_gate=gate)
    FakeRecognizer.script = [phrase("one"), phrase("two"), phrase("three")]
    run = Run(cache)
    assert run.wait(lambda: "mic" in run.events)           # capturing already ...
    assert run.wait(lambda: FakeRecognizer.listened >= 3)  # ... and all three phrases were captured
    assert run.events[0] == "model:loading" and not cache.is_ready("base")
    assert run.texts == []                                 # nothing can be transcribed yet
    gate.set()                                             # model finishes loading
    assert run.wait(lambda: len(run.texts) == 3)
    assert run.texts == ["one", "two", "three"]            # all kept, in order
    assert "model:ready" in run.events
    run.stop()


def test_listening_continues_while_a_clip_is_being_transcribed():
    gate = threading.Event()
    cache, _ = make_cache(model=FakeModel(gate=gate))
    FakeRecognizer.script = [phrase("a"), phrase("b"), phrase("c"), phrase("d")]
    run = Run(cache)
    assert run.wait(lambda: FakeRecognizer.listened >= 4)  # captured all four ...
    assert run.texts == []                                 # ... though the first is still transcribing
    gate.set()
    assert run.wait(lambda: len(run.texts) == 4)
    assert run.texts == ["a", "b", "c", "d"]
    run.stop()


def test_stop_returns_promptly_and_phrases_already_spoken_are_still_delivered():
    cache, _ = make_cache(model=FakeModel(delay=0.15))
    FakeRecognizer.script = [phrase("x1"), phrase("x2"), phrase("x3")]
    run = Run(cache)
    assert run.wait(lambda: FakeRecognizer.listened >= 3)
    t0 = time.time()
    run.stop()                                             # the capture loop ends ...
    assert time.time() - t0 < 2
    assert run.wait(lambda: len(run.texts) == 3)           # ... the last words still arrive
    assert run.texts == ["x1", "x2", "x3"]


def test_a_second_session_reuses_the_model_already_in_memory():
    model = FakeModel()
    cache, loads = make_cache(model=model)
    FakeRecognizer.script = [phrase("first")]
    one = Run(cache)
    assert one.wait(lambda: one.texts == ["first"])
    one.stop()

    FakeRecognizer.script = [phrase("second")]
    two = Run(cache)
    assert two.wait(lambda: two.texts == ["second"])
    assert two.events[0] == "model:ready"                  # no "loading" phase the second time
    assert loads == ["base"]                               # loaded ONCE for both sessions
    two.stop()


def test_listening_alone_keeps_the_model_from_being_unloaded():
    clock = types.SimpleNamespace(t=1000.0)
    cache = WhisperModelCache(loader=lambda n: FakeModel(), idle_unload_s=60, clock=lambda: clock.t)
    run = Run(cache)
    assert run.wait(lambda: "model:ready" in run.events and "mic" in run.events)
    clock.t += 10_000                                      # a long quiet stretch with the mic on
    assert cache.unload_if_idle() is False
    run.stop()


def test_a_model_that_cannot_load_is_reported_and_listening_stops():
    cache, _ = make_cache(fail=True)
    run = Run(cache)
    assert run.wait(lambda: run.errors)
    assert "no such model" in run.errors[0]
    assert run.wait(lambda: not run.runner.is_alive())     # capture loop ended by itself


def test_a_microphone_failure_is_reported_and_leaves_no_thread_behind(monkeypatch):
    def broken():
        raise OSError("no input device")
    monkeypatch.setattr(vi.sr, "Microphone", broken)
    cache, _ = make_cache()
    run = Run(cache)
    assert run.wait(lambda: run.errors) and "no input device" in run.errors[0]
    run.runner.join(5)
    assert not [t for t in threading.enumerate() if t.name == "doc-explainer-transcribe"]


def test_if_the_model_never_arrives_only_the_newest_phrases_are_kept():
    gate = threading.Event()
    cache, _ = make_cache(load_gate=gate)
    FakeRecognizer.script = [phrase(f"p{i}") for i in range(6)]
    run = Run(cache)
    run.thread.MAX_PENDING = 3
    assert run.wait(lambda: FakeRecognizer.listened >= 6)
    gate.set()
    assert run.wait(lambda: len(run.texts) >= 3)
    time.sleep(0.2)
    assert run.texts == ["p3", "p4", "p5"]
    run.stop()


# ----------------------------------------------------------------------------- the widget
class FakeThread(QObject):
    text_recognized = Signal(str)
    error_occurred = Signal(str)
    listening_started = Signal()
    model_state = Signal(str)
    notice = Signal(str)
    finished = Signal()
    instances = []

    def __init__(self, model_name="base", cache=None):
        super().__init__()
        self.model_name, self.started, self.stopped, self._finished = model_name, False, False, False
        FakeThread.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def isFinished(self):
        return self._finished

    def wait(self, *a):
        raise AssertionError("the UI thread must never wait() for the listening thread")


class FakeCache:
    def __init__(self, ready=False):
        self.ready, self.configured, self.loading = ready, [], []

    def is_ready(self, name=None):
        return self.ready

    def configure(self, idle_unload_s=None):
        self.configured.append(idle_unload_s)

    def start_loading(self, name):
        self.loading.append(name)


@pytest.fixture
def widget(qapp, monkeypatch):
    FakeThread.instances = []
    cache = FakeCache()
    monkeypatch.setattr(vi, "ListeningThread", FakeThread)
    monkeypatch.setattr(vi, "get_model_cache", lambda: cache)
    w = vi.VoiceInput()
    w.cache = cache
    return w


def test_status_only_says_listening_once_the_microphone_is_really_capturing(widget):
    widget.start_listening()
    assert "Starting microphone" in widget.status_label.text()      # not "Listening" yet
    assert widget.record_button.text().endswith("Stop Listening") and FakeThread.instances[0].started
    FakeThread.instances[0].listening_started.emit()
    assert "Listening" in widget.status_label.text() and "loading the speech model" in widget.status_label.text()
    FakeThread.instances[0].model_state.emit("ready")
    assert widget.status_label.text() == "Listening..."


def test_when_the_model_is_already_resident_there_is_no_loading_message(widget):
    widget.cache.ready = True
    widget.start_listening()
    FakeThread.instances[0].listening_started.emit()
    assert widget.status_label.text() == "Listening..."


def test_stop_never_blocks_and_the_thread_is_kept_until_it_has_finished(widget):
    widget.start_listening()
    thread = FakeThread.instances[0]
    widget.stop_listening()                                         # FakeThread.wait() would raise
    assert thread.stopped and not widget.is_recording and widget.listening_thread is None
    assert widget.record_button.text().endswith("Start Listening")
    assert thread in widget._finishing                              # still delivering the last words
    thread._finished = True
    thread.finished.emit()
    assert thread not in widget._finishing


def test_text_from_the_thread_reaches_the_rest_of_the_app(widget):
    got = []
    widget.voice_text.connect(got.append)
    widget.start_listening()
    FakeThread.instances[0].text_recognized.emit("hello there")
    assert got == ["hello there"]


def test_can_restart_immediately_after_stopping(widget):
    widget.start_listening(); widget.stop_listening(); widget.start_listening()
    assert len(FakeThread.instances) == 2 and widget.is_recording


def test_voice_config_sets_the_model_idle_time_and_can_preload(qapp, monkeypatch):
    cache = FakeCache()
    monkeypatch.setattr(vi, "get_model_cache", lambda: cache)
    w = vi.VoiceInput(voice_config=VoiceConfig(whisper_model="small", model_idle_unload_s=42, preload_model=True))
    assert cache.configured == [42] and cache.loading == ["small"] and w._model_name() == "small"
    cache2 = FakeCache()
    monkeypatch.setattr(vi, "get_model_cache", lambda: cache2)
    vi.VoiceInput(voice_config=VoiceConfig())                       # defaults: nothing loaded at start-up
    assert cache2.loading == [] and cache2.configured == [600.0]
