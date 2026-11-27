"""Opening a microphone must not spray ALSA/JACK diagnostics on the console.

Run in subprocesses: the noise is written by C libraries straight to the process's stderr, which
only a separate process can observe reliably.
"""
import subprocess
import sys

import pytest

NOISE = ("ALSA lib", "jack server is not running", "Cannot connect to server", "JackShmReadWritePtr")

PROBE = """
{silence}
import speech_recognition as sr
try:
    sr.Microphone()
    print("devices:", len(sr.Microphone.list_microphone_names()))
except Exception as e:
    print("probe error:", type(e).__name__)
"""
SILENCE = "from doc_explainer_ui.gui.utils.audio_noise import silence_audio_library_noise; silence_audio_library_noise()"


def run(code, env_extra=None):
    import os
    env = {**os.environ, **(env_extra or {})}
    env.pop("DOC_EXPLAINER_AUDIO_DEBUG", None) if not env_extra else None
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)


def noise_lines(proc):
    return [line for line in proc.stderr.splitlines() if any(n in line for n in NOISE)]


@pytest.fixture(scope="module")
def baseline():
    proc = run(PROBE.format(silence=""))
    if not noise_lines(proc):
        pytest.skip("this machine's audio libraries print nothing, so there is nothing to silence")
    return proc


def test_noise_is_gone_but_devices_are_still_found(baseline):
    fixed = run(PROBE.format(silence=SILENCE))
    assert noise_lines(fixed) == []
    assert fixed.returncode == 0
    assert fixed.stdout.strip().split("\n")[-1] == baseline.stdout.strip().split("\n")[-1]   # same device count


def test_debug_variable_keeps_the_messages(baseline):
    loud = run(PROBE.format(silence=SILENCE), env_extra={"DOC_EXPLAINER_AUDIO_DEBUG": "1"})
    assert noise_lines(loud)


def test_installing_twice_and_without_the_libraries_is_harmless(monkeypatch):
    from doc_explainer_ui.gui.utils import audio_noise as an
    monkeypatch.setattr(an, "_installed", False)
    monkeypatch.setattr(an, "_load", lambda *names: None)         # neither ALSA nor JACK installed
    assert an.silence_audio_library_noise() is False
    assert an.silence_audio_library_noise() is False
