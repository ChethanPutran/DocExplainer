"""UI configuration: defaults live in the packaged YAML only; the user file and environment override them."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from doc_explainer_ui import config as cfgmod
from doc_explainer_ui.bootstrap import ensure_default_configs
from doc_explainer_ui.config import UIConfig, packaged_defaults
from doc_explainer_ui.gui.windows.settings_window import SettingsWindow

SRC = Path(cfgmod.__file__).parent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("DOC_EXPLAINER_UI_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cfgmod, "CONFIG_FILE", tmp_path / "nohome" / "ui_config.yaml")


def write(tmp_path, text):
    f = tmp_path / "ui.yaml"
    f.write_text(text)
    return str(f)


def test_dataclass_defaults_ARE_the_packaged_yaml():
    d, c = packaged_defaults(), UIConfig()
    for section, values in d.items():
        assert {k: getattr(getattr(c, section), k) for k in values} == values, section


def test_every_setting_in_the_yaml_has_a_field_and_the_other_way_round():
    d, c = packaged_defaults(), UIConfig()
    assert set(d) == set(UIConfig.__dataclass_fields__)
    for section, values in d.items():
        assert set(values) == set(type(getattr(c, section)).__dataclass_fields__), section


def test_a_user_file_may_hold_just_the_keys_it_changes(tmp_path):
    c = UIConfig.load(write(tmp_path, "voice:\n  whisper_model: small\n"))
    assert c.voice.whisper_model == "small" and c.voice.enabled is True and c.window.width == 1200


def test_environment_overrides_file_and_defaults(tmp_path, monkeypatch):
    f = write(tmp_path, "server:\n  url: http://file:1\n")
    monkeypatch.setenv("DOC_EXPLAINER_UI_SERVER_URL", "http://env:2")
    monkeypatch.setenv("DOC_EXPLAINER_UI_VOICE_PRELOAD_MODEL", "true")
    monkeypatch.setenv("DOC_EXPLAINER_UI_SERVER_TIMEOUT_S", "30")
    c = UIConfig.load(f)
    assert (c.server.url, c.voice.preload_model, c.server.timeout_s) == ("http://env:2", True, 30)


def test_missing_files(tmp_path):
    assert UIConfig.load().server.protocol == "rest"                       # default location: not needed
    with pytest.raises(FileNotFoundError):
        UIConfig.load(str(tmp_path / "nope.yaml"))


def test_unreadable_old_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="Move it aside"):
        UIConfig.load(write(tmp_path, "a: !!python/object:os.system {}\n"))


def test_unknown_keys_are_ignored(tmp_path):
    assert UIConfig.load(write(tmp_path, "window:\n  width: 900\n  colour: red\nllm:\n  provider: x\n")).window.width == 900


def test_save_writes_only_differences_and_round_trips(tmp_path):
    f = tmp_path / "out.yaml"
    c = UIConfig()
    c.theme.font_size = 14
    c.save(str(f))
    assert yaml.safe_load(f.read_text()) == {"theme": {"font_size": 14}}
    assert UIConfig.load(str(f)).theme.font_size == 14


def test_bootstrap_creates_a_commented_override_file(tmp_path):
    created = ensure_default_configs(tmp_path)
    assert [f.name for f in created] == ["ui_config.yaml"]
    text = created[0].read_text()
    assert yaml.safe_load(text) is None and all(l.startswith("#") or not l.strip() for l in text.splitlines())
    (tmp_path / "ui_config.yaml").write_text("window:\n  width: 700\n")
    assert ensure_default_configs(tmp_path) == [] and UIConfig.load(str(tmp_path / "ui_config.yaml")).window.width == 700


def test_there_is_no_second_ui_config_class_or_qsettings_store():
    assert not (SRC / "gui" / "config.py").exists()
    for f in SRC.rglob("*.py"):
        assert "QSettings" not in f.read_text(), f


def test_no_default_value_is_repeated_in_code():
    """Spot-check: literals that are YAML defaults must not be re-typed as defaults in the UI code."""
    needles = ("http://127.0.0.1:8000", "~/Documents", "~/.doc_explainer/cache")
    hits = [(f.name, n) for f in SRC.rglob("*.py") if f.name != "bootstrap.py"
            for n in needles if n in f.read_text() and f.name != "config.py"]
    assert hits == [], hits


def test_the_settings_window_edits_the_config_and_saves_only_the_difference(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "CONFIG_FILE", tmp_path / "ui_config.yaml")
    themes = []
    tm = SimpleNamespace(get_theme_names=lambda: ["light", "dark"], set_theme=themes.append)
    config = UIConfig()
    w = SettingsWindow(config, tm)
    assert w.whisper.currentText() == "base" and w.protocol.currentText() == "rest"
    w.whisper.setCurrentText("small")
    w.server_url.setText("http://srv:9")
    w.theme.setCurrentText("dark")
    assert themes[-1] == "dark"                                            # live preview
    w._save()
    assert (config.voice.whisper_model, config.server.url) == ("small", "http://srv:9")
    assert yaml.safe_load((tmp_path / "ui_config.yaml").read_text()) == {
        "theme": {"name": "dark"}, "voice": {"whisper_model": "small"}, "server": {"url": "http://srv:9"}}
