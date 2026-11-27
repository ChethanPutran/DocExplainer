"""The resident Whisper model: loaded once, shared, kept while in use, unloaded when idle."""
import sys
import threading
import time
import types

import pytest

from doc_explainer_ui.gui.widgets.voice import speech_model as sm
from doc_explainer_ui.gui.widgets.voice.speech_model import ModelLoadError, WhisperModelCache


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(loader=None, idle=600.0, clock=None, **kw):
    loads = []

    def default(name):
        loads.append(name)
        return types.SimpleNamespace(name=name)

    def counting(name):
        loads.append(name)
        return loader(name)

    cache = WhisperModelCache(loader=counting if loader else default, idle_unload_s=idle,
                              clock=clock or time.monotonic, **kw)
    return cache, loads


def test_start_loading_returns_at_once_and_the_model_loads_only_once():
    gate = threading.Event()
    cache, loads = make(loader=lambda n: gate.wait(5) and types.SimpleNamespace(name=n))
    t0 = time.time()
    for _ in range(5):
        cache.start_loading("base")                       # repeated starts must not reload
    assert time.time() - t0 < 0.5 and cache.state == "loading" and not cache.is_ready("base")
    gate.set()
    cache.wait_ready("base", timeout=5)
    assert cache.is_ready("base") and loads == ["base"]


def test_every_use_shares_the_same_resident_model():
    cache, loads = make()
    with cache.use("base") as a:
        pass
    with cache.use("base") as b:
        pass
    cache.start_loading("base")                           # already resident: a no-op, not a reload
    assert a is b and loads == ["base"]


def test_use_waits_for_a_model_that_is_still_loading():
    gate = threading.Event()
    cache, _ = make(loader=lambda n: gate.wait(5) and types.SimpleNamespace(name=n))
    cache.start_loading("base")
    got = []
    t = threading.Thread(target=lambda: got.append(cache.use("base").__enter__().name))
    t.start()
    time.sleep(0.1)
    assert got == []                                      # blocked while loading
    gate.set()
    t.join(5)
    assert got == ["base"]


def test_transcriptions_never_overlap_on_the_shared_model():
    cache, _ = make()
    inside, peak, lock = [0], [0], threading.Lock()

    def work():
        with cache.use("base"):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.02)
            with lock:
                inside[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert peak[0] == 1


# ---------------------------------------------------------------------------- idle unload
def test_unloads_after_the_idle_period_and_reloads_on_next_use():
    clock = Clock()
    cache, loads = make(idle=600, clock=clock)
    with cache.use("base"):
        pass
    clock.t += 599
    assert cache.unload_if_idle() is False and cache.state == "ready"
    clock.t += 2
    assert cache.unload_if_idle() is True and cache.state == "unloaded" and not cache.is_ready()
    with cache.use("base") as m:                          # asking again just loads it again
        assert m.name == "base"
    assert loads == ["base", "base"] and cache.state == "ready"


def test_activity_restarts_the_idle_clock():
    clock = Clock()
    cache, _ = make(idle=600, clock=clock)
    with cache.use("base"):
        pass
    clock.t += 500
    with cache.use("base"):                               # used again at +500
        pass
    clock.t += 500                                        # +1000 overall, but only 500 since last use
    assert cache.unload_if_idle() is False
    clock.t += 101
    assert cache.unload_if_idle() is True


def test_never_unloaded_while_transcribing_or_while_the_microphone_is_on():
    clock = Clock()
    cache, _ = make(idle=60, clock=clock)
    with cache.use("base"):
        clock.t += 10_000
        assert cache.unload_if_idle() is False            # a transcription is in flight
    clock.t += 10_000
    with cache.session():
        clock.t += 10_000
        assert cache.unload_if_idle() is False            # listening counts as activity
    assert cache.unload_if_idle() is False                # idle clock restarted when the session ended
    clock.t += 61
    assert cache.unload_if_idle() is True


def test_idle_zero_keeps_the_model_for_good():
    clock = Clock()
    cache, _ = make(idle=0, clock=clock)
    with cache.use("base"):
        pass
    clock.t += 10 ** 9
    assert cache.unload_if_idle() is False and cache.state == "ready"


def test_configure_changes_the_idle_period():
    clock = Clock()
    cache, _ = make(idle=600, clock=clock)
    with cache.use("base"):
        pass
    cache.configure(idle_unload_s=30)
    clock.t += 31
    assert cache.unload_if_idle() is True


def test_a_background_watcher_unloads_by_itself():
    cache, _ = make(idle=0.15, poll_s=0.02)
    with cache.use("base"):
        pass
    end = time.time() + 3
    while cache.state != "unloaded" and time.time() < end:
        time.sleep(0.02)
    assert cache.state == "unloaded"
    cache.close()


# ---------------------------------------------------------------------------- failure / switching
def test_a_failed_load_is_reported_to_waiters_and_can_be_retried():
    attempts = []

    def flaky(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("disk on fire")
        return types.SimpleNamespace(name=name)

    cache, _ = make(loader=flaky)
    with pytest.raises(ModelLoadError, match="disk on fire"):
        cache.wait_ready("base", timeout=5)
    assert cache.state == "unloaded"
    cache.wait_ready("base", timeout=5)                   # next request tries again and succeeds
    assert cache.is_ready("base") and len(attempts) == 2


def test_asking_for_a_different_model_replaces_the_resident_one():
    cache, loads = make()
    with cache.use("base") as a:
        assert a.name == "base"
    with cache.use("small") as b:
        assert b.name == "small"
    assert loads == ["base", "small"] and cache.is_ready("small") and not cache.is_ready("base")


def test_a_superseded_load_cannot_overwrite_the_newer_one():
    gate = threading.Event()

    def loader(name):
        if name == "slow":
            gate.wait(5)
        return types.SimpleNamespace(name=name)

    cache, _ = make(loader=loader)
    cache.start_loading("slow")
    cache.wait_ready("fast", timeout=5)                   # asked for another while 'slow' was loading
    gate.set()
    time.sleep(0.2)                                       # let the stale 'slow' load finish
    with cache.use("fast") as m:
        assert m.name == "fast"


def test_wait_ready_times_out():
    gate = threading.Event()
    cache, _ = make(loader=lambda n: gate.wait(5) and types.SimpleNamespace(name=n))
    with pytest.raises(TimeoutError):
        cache.wait_ready("base", timeout=0.1)
    gate.set()


# ---------------------------------------------------------------------------- default loader
def test_default_loader_loads_on_the_device_pick_device_chose(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(
        load_model=lambda name, device=None: seen.update(name=name, device=device) or "model"))
    monkeypatch.setattr(sm, "pick_device", lambda: "cpu")
    assert sm._default_loader("base") == "model"
    assert seen == {"name": "base", "device": "cpu"}
