"""Configuration has ONE source for every default (the packaged YAML); files and environment only override."""
import ast
import logging
from pathlib import Path

import pytest
import yaml

from doc_explainer.api.factory import llm_kwargs
from doc_explainer.config import BackendConfig, LLMConfig
from doc_explainer.config.bootstrap import ensure_default_configs
from doc_explainer.config.layered import deep_merge, load_layered, packaged_defaults
from doc_explainer.orchestrator import SystemConfig

LING = "inclusionai/ling-3.0-flash"
SRC = Path(__file__).resolve().parents[1] / "src" / "doc_explainer"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    import os
    for k in list(os.environ):
        if k.startswith("DOC_EXPLAINER_") or k.endswith("_API_KEY"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("doc_explainer.config.layered.CONFIG_DIR", tmp_path / "nohome")   # never read the real ~/.doc_explainer


def write(tmp_path, text, name="llm_config.yaml"):
    f = tmp_path / name
    f.write_text(text)
    return str(f)


# ----------------------------------------------------------------------------- the defaults come from the YAML
def test_dataclass_defaults_ARE_the_packaged_yaml():
    d = packaged_defaults("llm_config")
    cfg = LLMConfig()
    assert {k: getattr(cfg, k) for k in d["llm"]} == d["llm"] and cfg.providers == d["providers"]
    b, y = BackendConfig(), packaged_defaults("backend_config")
    assert b.logging.level == y["logging"]["level"] and b.embeddings.model == y["embeddings"]["model"]


def test_the_wanted_defaults():
    cfg = LLMConfig.load()                                    # no user file at all
    assert (cfg.provider, cfg.resolved_model, cfg.requests_per_minute) == ("airouter", LING, 10)
    assert cfg.resolved_base_url == "https://api.airouter.in/v1"


def _code_strings(path):
    """String literals in code (docstrings excluded: they may mention a model by name)."""
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_no_default_is_written_anywhere_else_in_the_code():
    """Model names and endpoints appear in the YAML only (a copy in code is how they drift apart)."""
    needles = ("ling-3.0-flash", "api.airouter.in", "kimi-k3", "gpt-4", "gemini-3", "llama3", "all-MiniLM", "localhost:11434", "openrouter.ai")
    offenders = [(f.name, n) for f in SRC.rglob("*.py") for lit in _code_strings(f) for n in needles if n in lit]
    assert offenders == [], offenders


# ----------------------------------------------------------------------------- the user file overrides only what it names
def test_a_user_file_may_hold_just_the_keys_it_changes(tmp_path):
    cfg = LLMConfig.load(write(tmp_path, "llm:\n  requests_per_minute: 3\n"))
    assert cfg.requests_per_minute == 3 and cfg.provider == "airouter" and cfg.resolved_model == LING


def test_switching_provider_alone_uses_that_providers_own_model_and_endpoint(tmp_path):
    cfg = LLMConfig.load(write(tmp_path, "llm:\n  provider: openrouter\n"))
    assert cfg.resolved_model == "moonshotai/kimi-k3" and cfg.api_key is None
    assert LLMConfig.load(write(tmp_path, "llm:\n  provider: gemini\n")).resolved_model == "gemini-3.5-flash"


def test_an_explicit_model_or_endpoint_wins(tmp_path):
    cfg = LLMConfig.load(write(tmp_path, "llm:\n  model: google/gemini-3-flash\n  base_url: http://proxy:9/v1\n"))
    assert cfg.resolved_model == "google/gemini-3-flash" and cfg.resolved_base_url == "http://proxy:9/v1"


def test_a_provider_table_entry_can_change_that_providers_default(tmp_path):
    cfg = LLMConfig.load(write(tmp_path, "providers:\n  airouter:\n    model: x/y\n"))
    assert cfg.resolved_model == "x/y" and cfg.resolved_base_url == "https://api.airouter.in/v1"   # the rest is kept


def test_a_missing_explicit_file_is_an_error_but_a_missing_default_file_is_not(tmp_path):
    with pytest.raises(FileNotFoundError):
        LLMConfig.load(str(tmp_path / "nope.yaml"))
    assert LLMConfig.load().provider == "airouter"


def test_keys_from_older_files_are_ignored_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = BackendConfig.load(write(tmp_path, "memory:\n  enabled: true\nlogging:\n  level: debug\n  colour: red\n", "backend_config.yaml"))
    assert cfg.logging.level == "DEBUG" and "colour" in caplog.text


# ----------------------------------------------------------------------------- environment
def test_environment_overrides_file_and_defaults(tmp_path, monkeypatch):
    f = write(tmp_path, "llm:\n  requests_per_minute: 3\n")
    monkeypatch.setenv("DOC_EXPLAINER_LLM_REQUESTS_PER_MINUTE", "20")
    monkeypatch.setenv("DOC_EXPLAINER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DOC_EXPLAINER_LLM_REASONING", "false")
    monkeypatch.setenv("DOC_EXPLAINER_LLM_MODEL", "gpt-5")
    cfg = LLMConfig.load(f)
    assert (cfg.requests_per_minute, cfg.provider, cfg.reasoning, cfg.resolved_model) == (20, "openai", False, "gpt-5")
    monkeypatch.setenv("DOC_EXPLAINER_LLM_MODEL", "null")
    assert LLMConfig.load(f).resolved_model == "gpt-4o-mini"                 # null = the provider's default again


def test_environment_reaches_the_backend_sections_and_provider_table(monkeypatch):
    monkeypatch.setenv("DOC_EXPLAINER_LOGGING_LEVEL", "WARNING")
    monkeypatch.setenv("DOC_EXPLAINER_KNOWLEDGE_GRAPH_ENABLED", "false")
    monkeypatch.setenv("DOC_EXPLAINER_EMBEDDINGS_MODEL", "other-model")
    monkeypatch.setenv("DOC_EXPLAINER_PROVIDERS_AIROUTER_MODEL", "from/env")
    b = BackendConfig.load()
    assert (b.logging.level, b.knowledge_graph.enabled, b.embeddings.model) == ("WARNING", False, "other-model")
    assert LLMConfig.load().resolved_model == "from/env"


def test_keys_come_from_the_environment_named_in_the_provider_table(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "air")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    assert LLMConfig(provider="airouter").api_key == "air" and LLMConfig(provider="AIRouter").api_key == "air"
    assert LLMConfig(provider="openrouter").api_key == "or" and LLMConfig(provider="ollama").api_key is None
    monkeypatch.delenv("AIROUTER_API_KEY")
    with pytest.raises(ValueError, match="AIROUTER_API_KEY"):
        LLMConfig().validate()


def test_validate_rejects_nonsense():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMConfig(provider="nonsense").validate()
    with pytest.raises(ValueError, match="requests_per_minute"):
        LLMConfig(provider="ollama", requests_per_minute=0).validate()
    with pytest.raises(ValueError, match="logging level"):
        b = BackendConfig(); b.logging.level = "LOUD"; b.validate()


def test_what_the_wrappers_are_given(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "air-key")
    air = llm_kwargs(LLMConfig(max_tokens=321))
    assert air["api_key"] == "air-key" and air["base_url"] == "https://api.airouter.in/v1"
    assert air["max_tokens"] == 321 and air["temperature"] == 1.0 and "model_name" not in air
    assert "api_key" not in llm_kwargs(LLMConfig(provider="openrouter"))
    assert llm_kwargs(LLMConfig(provider="ollama"))["base_url"] == "http://localhost:11434"


def test_the_system_has_no_model_defaults_of_its_own(tmp_path):
    from doc_explainer.orchestrator import System
    from fakes import HashingEmbedder, ScriptedLLM
    with pytest.raises(ValueError, match="llm_provider and llm_model"):
        System(SystemConfig(data_dir=tmp_path), embedder=HashingEmbedder())
    with pytest.raises(ValueError, match="embedding_model"):
        System(SystemConfig(data_dir=tmp_path), llm=object())


def test_bootstrap_creates_commented_override_files_not_copies_of_the_defaults(tmp_path):
    created = ensure_default_configs(tmp_path)
    assert sorted(f.name for f in created) == ["backend_config.yaml", "llm_config.yaml"]
    for f in created:
        text = f.read_text()
        assert all(line.startswith("#") or not line.strip() for line in text.splitlines())   # nothing active
        assert yaml.safe_load(text) is None
    assert LLMConfig.load(str(tmp_path / "llm_config.yaml")).resolved_model == LING          # so the defaults still apply
    (tmp_path / "llm_config.yaml").write_text("llm:\n  provider: openai\n")
    assert ensure_default_configs(tmp_path) == [] and "openai" in (tmp_path / "llm_config.yaml").read_text()   # never overwritten


def test_save_writes_only_the_differences(tmp_path):
    f = tmp_path / "llm.yaml"
    cfg = LLMConfig(requests_per_minute=7)
    cfg.save(str(f))
    assert yaml.safe_load(f.read_text()) == {"llm": {"requests_per_minute": 7}}
    assert LLMConfig.load(str(f)).requests_per_minute == 7
    b = tmp_path / "b.yaml"
    c = BackendConfig(); c.logging.level = "ERROR"
    c.save(str(b))
    assert yaml.safe_load(b.read_text()) == {"logging": {"level": "ERROR"}}


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3} and base["a"]["b"] == 1


def test_a_dotenv_file_in_the_working_directory_supplies_the_api_key(tmp_path, monkeypatch):
    import importlib
    import doc_explainer.config.layered as layered
    (tmp_path / ".env").write_text("AIROUTER_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    importlib.reload(layered)                                   # the .env is read when the config module loads
    assert LLMConfig().api_key                                  # found in a .env (the project's, else this one)
    LLMConfig().validate()                                      # so the server starts without exporting it by hand


def test_the_token_budget_leaves_room_for_a_reasoning_models_thinking():
    """Measured: with 1000 the default model hit finish_reason=length on a section-sized prompt (empty JSON);
    the same prompt needed ~1800 tokens in total."""
    assert LLMConfig().max_tokens >= 4000
