"""Voice input must not try to use a GPU this PyTorch build cannot run on."""
import types

import pytest

from doc_explainer_ui.gui.widgets.voice import voice_input as vi


def fake_torch(monkeypatch, available=True, cc=(6, 1), archs=("sm_75", "sm_80", "sm_90", "compute_120")):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda: cc,
        get_arch_list=lambda: list(archs),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", types.SimpleNamespace(cuda=cuda))


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("DOC_EXPLAINER_WHISPER_DEVICE", raising=False)


def test_old_gpu_without_kernels_falls_back_to_cpu(monkeypatch):   # the reported MX350 case
    fake_torch(monkeypatch, cc=(6, 1))
    assert vi.pick_device() == "cpu"


def test_no_gpu_means_cpu(monkeypatch):
    fake_torch(monkeypatch, available=False)
    assert vi.pick_device() == "cpu"


def test_supported_gpu_is_used(monkeypatch):
    fake_torch(monkeypatch, cc=(8, 6), archs=("sm_75", "sm_86"))
    assert vi.pick_device() == "cuda"


def test_newer_gpu_can_use_ptx_forward_compatibility(monkeypatch):
    fake_torch(monkeypatch, cc=(8, 9), archs=("sm_75", "sm_86", "compute_86"))
    assert vi.pick_device() == "cuda"
    fake_torch(monkeypatch, cc=(6, 1), archs=("sm_75", "compute_75"))   # PTX needs cc >= its target
    assert vi.pick_device() == "cpu"


def test_probing_errors_fall_back_to_cpu(monkeypatch):
    fake_torch(monkeypatch)
    monkeypatch.setattr(__import__("sys").modules["torch"].cuda, "get_device_capability",
                        lambda: (_ for _ in ()).throw(RuntimeError("driver exploded")))
    assert vi.pick_device() == "cpu"


def test_environment_override_wins(monkeypatch):
    fake_torch(monkeypatch, cc=(6, 1))
    monkeypatch.setenv("DOC_EXPLAINER_WHISPER_DEVICE", "cuda")
    assert vi.pick_device() == "cuda"
    monkeypatch.setenv("DOC_EXPLAINER_WHISPER_DEVICE", "nonsense")
    assert vi.pick_device() == "cpu"          # ignored, normal detection applies


def test_audio_stats_reports_real_duration_and_level():
    import math
    import struct

    import speech_recognition as sr

    rate = 16000
    tone = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(rate * 2))
    seconds, rms, peak = vi.audio_stats(sr.AudioData(tone, rate, 2))
    assert seconds == pytest.approx(2.0)                 # NOT 4.0: samples are 2 bytes wide
    assert rms == pytest.approx(8000 / math.sqrt(2), rel=0.02) and peak == pytest.approx(8000, rel=0.01)

    silence = vi.audio_stats(sr.AudioData(b"\x00\x00" * rate, rate, 2))
    assert silence[0] == pytest.approx(1.0) and silence[1] == 0 and silence[2] == 0   # a muted mic looks like this
