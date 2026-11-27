"""Backend service settings. Every default lives in ``defaults/backend_config.yaml`` (see ``layered``)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .layered import CONFIG_DIR, difference, known_keys_only, load_layered, packaged_defaults

_D = packaged_defaults("backend_config")
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class KnowledgeGraphConfig:
    enabled: bool = _D["knowledge_graph"]["enabled"]


@dataclass
class EmbeddingConfig:
    model: str = _D["embeddings"]["model"]
    check_for_updates: bool = _D["embeddings"]["check_for_updates"]


@dataclass
class LoggingConfig:
    level: str = _D["logging"]["level"]
    log_directory: str = _D["logging"]["log_directory"]
    log_file: str = _D["logging"]["log_file"]
    http_log_file: str = _D["logging"]["http_log_file"]
    quiet_level: str = _D["logging"]["quiet_level"]
    quiet: list = field(default_factory=lambda: list(_D["logging"]["quiet"]))
    console_enabled: bool = _D["logging"]["console_enabled"]
    file_enabled: bool = _D["logging"]["file_enabled"]
    max_file_size_mb: int = _D["logging"]["max_file_size_mb"]
    backup_count: int = _D["logging"]["backup_count"]


@dataclass
class BackendConfig:
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "BackendConfig":
        """defaults <- user file (``filepath`` or ``~/.doc_explainer/config/backend_config.yaml``) <- environment."""
        data = load_layered("backend_config", filepath)
        cfg = cls(
            knowledge_graph=KnowledgeGraphConfig(**known_keys_only(KnowledgeGraphConfig, data.get("knowledge_graph"))),
            embeddings=EmbeddingConfig(**known_keys_only(EmbeddingConfig, data.get("embeddings"))),
            logging=LoggingConfig(**known_keys_only(LoggingConfig, data.get("logging"))),
        )
        cfg.logging.level = str(cfg.logging.level).upper()
        return cfg

    def validate(self) -> None:
        if self.logging.level not in _LOG_LEVELS:
            raise ValueError(f"Invalid logging level: {self.logging.level}. Expected one of {sorted(_LOG_LEVELS)}.")
        if self.logging.max_file_size_mb <= 0:
            raise ValueError("logging.max_file_size_mb must be greater than 0.")
        if self.logging.backup_count < 0:
            raise ValueError("logging.backup_count cannot be negative.")
        if not self.embeddings.model:
            raise ValueError("embeddings.model cannot be empty.")

    @property
    def log_directory(self) -> Path:
        return Path(self.logging.log_directory).expanduser()

    @property
    def log_file(self) -> Path:
        return self.log_directory / self.logging.log_file

    @property
    def http_log_file(self) -> Path:
        return self.log_directory / self.logging.http_log_file

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath: Optional[str] = None) -> None:
        """Write only what differs from the defaults."""
        path = Path(filepath).expanduser() if filepath else CONFIG_DIR / "backend_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(difference(self.to_dict(), packaged_defaults("backend_config")), sort_keys=False), encoding="utf-8")
