"""Build the real system from the server's own config (the UI no longer does this)."""
from __future__ import annotations
import logging

from doc_explainer.config import BackendConfig, LLMConfig
from doc_explainer.config.bootstrap import ensure_default_configs

from .http_log import setup_http_logging
from .settings import ApiSettings

logger = logging.getLogger(__name__)


def llm_kwargs(llm: LLMConfig) -> dict:
    """Keyword arguments every LLM wrapper is created with, from the server's LLM config."""
    return {
        "temperature": llm.temperature, "max_tokens": llm.max_tokens, "max_tokens_cap": llm.max_tokens_cap,
        "token_growth": llm.token_growth, "token_retries": llm.token_retries, "timeout": llm.timeout,
        "requests_per_minute": llm.requests_per_minute,
        "min_request_interval_seconds": llm.min_request_interval_seconds,
        "rate_limit_retries": llm.rate_limit_retries,
        **llm.client_kwargs(),  # e.g. AIRouter's key and endpoint
    }


def build_system(settings: ApiSettings):
    """The real system, from the server's own config. Heavy: loads the embedding model on first use."""
    from doc_explainer.config.logger import setup_logging
    from doc_explainer.orchestrator import SystemConfig, build_system as _build

    if not (settings.llm_config_path and settings.backend_config_path):
        try:  # first run: create the (commented) override files in ~/.doc_explainer/config
            ensure_default_configs()
        except OSError:
            logger.warning("Could not create the config files; using the packaged defaults")

    llm = LLMConfig.load(settings.llm_config_path)          # packaged defaults <- user file <- environment
    backend = BackendConfig.load(settings.backend_config_path)
    llm.validate()
    backend.validate()
    try:
        setup_logging(backend)
        setup_http_logging(backend.http_log_file, max_bytes=backend.logging.max_file_size_mb * 1024 * 1024,
                           backups=backend.logging.backup_count)
    except Exception:
        logger.exception("Could not apply the logging configuration")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return _build(SystemConfig(
        data_dir=settings.data_dir / "system", llm_provider=llm.provider, llm_model=llm.resolved_model,
        llm_kwargs=llm_kwargs(llm),
        embedding_model=backend.embeddings.model, embedding_check_for_updates=backend.embeddings.check_for_updates,
        knowledge_enabled=backend.knowledge_graph.enabled))
