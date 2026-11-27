"""Starting the server must not ask HuggingFace about a model that is already on disk, nor flood the log."""
import logging

import pytest

from doc_explainer.config import BackendConfig
from doc_explainer.config.logger import setup_logging
from doc_explainer.core.indexing.embedder import SentenceTransformerEmbedder


class Factory:
    """Stands in for SentenceTransformer: records how it was asked to load."""

    def __init__(self, cached=True):
        self.calls, self.cached = [], cached

    def __call__(self, name, device=None, local_files_only=False):
        self.calls.append(local_files_only)
        if local_files_only and not self.cached:
            raise OSError("not in the local cache")
        return object()


def test_a_cached_model_is_loaded_without_any_network_call():
    f = Factory(cached=True)
    SentenceTransformerEmbedder("m")._open(f, "cpu")
    assert f.calls == [True]                                              # one attempt, local files only


def test_a_model_that_is_not_downloaded_yet_is_fetched_once():
    f = Factory(cached=False)
    SentenceTransformerEmbedder("m")._open(f, "cpu")
    assert f.calls == [True, False]                                       # tried the disk first, then downloaded


def test_asking_for_update_checks_goes_online_directly():
    f = Factory(cached=True)
    SentenceTransformerEmbedder("m", check_for_updates=True)._open(f, "cpu")
    assert f.calls == [False]


def test_the_real_model_loads_from_the_local_copy_without_touching_the_hub(monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")                              # any network attempt would now fail loudly
    try:
        e = SentenceTransformerEmbedder("all-MiniLM-L6-v2", device="cpu")
        assert e.dimension == 384
    except OSError:
        pytest.skip("the model is not cached on this machine")


def test_the_default_is_to_not_check_for_updates_and_it_is_configurable(monkeypatch):
    assert BackendConfig().embeddings.check_for_updates is False
    monkeypatch.setenv("DOC_EXPLAINER_EMBEDDINGS_CHECK_FOR_UPDATES", "true")
    assert BackendConfig.load().embeddings.check_for_updates is True


def test_chatty_libraries_are_hidden_below_warning_and_the_list_is_configurable(tmp_path, monkeypatch):
    cfg = BackendConfig()
    cfg.logging.log_directory = str(tmp_path)
    cfg.logging.console_enabled = False
    names = ["httpx", "httpcore", "huggingface_hub", "sentence_transformers", "transformers", "urllib3", "chromadb", "openai"]
    for n in names:
        logging.getLogger(n).setLevel(logging.NOTSET)
    setup_logging(cfg)
    assert all(logging.getLogger(n).level == logging.WARNING for n in names)
    logging.getLogger("httpx").info("HTTP Request: HEAD https://huggingface.co/...")          # not recorded
    logging.getLogger("doc_explainer.x").info("our own message")                               # still recorded
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "app.log").read_text()
    assert "our own message" in text and "HTTP Request" not in text
    monkeypatch.setenv("DOC_EXPLAINER_LOGGING_QUIET_LEVEL", "ERROR")
    assert BackendConfig.load().logging.quiet_level == "ERROR"
    logging.basicConfig(force=True)
