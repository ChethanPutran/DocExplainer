from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class KnowledgeGraphConfig:
    """Knowledge graph configuration."""

    enabled: bool = True
    auto_build: bool = True


@dataclass
class MemoryConfig:
    """Memory and session tracking configuration."""

    enabled: bool = True
    session_tracking: bool = True


@dataclass
class RetrievalConfig:
    """Document retrieval configuration."""

    enabled: bool = True
    top_k: int = 5


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    provider: str = "default"
    model: str = ""


@dataclass
class StorageConfig:
    """Backend storage configuration."""

    persist_directory: str = "~/.doc_explainer/cache"


@dataclass
class LoggingConfig:
    """Application logging configuration."""

    level: str = "INFO"
    log_directory: str = "~/.doc_explainer/logs"
    log_file: str = "app.log"
    console_enabled: bool = True
    file_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class BackendConfig:
    """Configuration for backend application services."""

    knowledge_graph: KnowledgeGraphConfig = field(
        default_factory=KnowledgeGraphConfig
    )

    memory: MemoryConfig = field(
        default_factory=MemoryConfig
    )

    retrieval: RetrievalConfig = field(
        default_factory=RetrievalConfig
    )

    embeddings: EmbeddingConfig = field(
        default_factory=EmbeddingConfig
    )

    storage: StorageConfig = field(
        default_factory=StorageConfig
    )

    logging: LoggingConfig = field(
        default_factory=LoggingConfig
    )

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "BackendConfig":
        """Load backend configuration from YAML."""

        if filepath is None:
            filepath = str(
                Path.home()
                / ".doc_explainer"
                / "config"
                / "backend_config.yaml"
            )

        path = Path(filepath).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"Backend configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        kg_data = data.get("knowledge_graph", {})
        memory_data = data.get("memory", {})
        retrieval_data = data.get("retrieval", {})
        embedding_data = data.get("embeddings", {})
        storage_data = data.get("storage", {})
        logging_data = data.get("logging", {})

        return cls(
            knowledge_graph=KnowledgeGraphConfig(
                enabled=kg_data.get("enabled", True),
                auto_build=kg_data.get("auto_build", True),
            ),

            memory=MemoryConfig(
                enabled=memory_data.get("enabled", True),
                session_tracking=memory_data.get(
                    "session_tracking",
                    True,
                ),
            ),

            retrieval=RetrievalConfig(
                enabled=retrieval_data.get("enabled", True),
                top_k=retrieval_data.get("top_k", 5),
            ),

            embeddings=EmbeddingConfig(
                provider=embedding_data.get(
                    "provider",
                    "default",
                ),
                model=embedding_data.get(
                    "model",
                    "",
                ),
            ),

            storage=StorageConfig(
                persist_directory=storage_data.get(
                    "persist_directory",
                    "~/.doc_explainer/cache",
                ),
            ),

            logging=LoggingConfig(
                level=logging_data.get(
                    "level",
                    "INFO",
                ).upper(),

                log_directory=logging_data.get(
                    "log_directory",
                    "~/.doc_explainer/logs",
                ),

                log_file=logging_data.get(
                    "log_file",
                    "app.log",
                ),

                console_enabled=logging_data.get(
                    "console_enabled",
                    True,
                ),

                file_enabled=logging_data.get(
                    "file_enabled",
                    True,
                ),

                max_file_size_mb=logging_data.get(
                    "max_file_size_mb",
                    10,
                ),

                backup_count=logging_data.get(
                    "backup_count",
                    5,
                ),
            ),
        )

    def validate(self) -> None:
        """Validate backend configuration."""

        if self.retrieval.top_k <= 0:
            raise ValueError(
                "retrieval.top_k must be greater than 0."
            )

        if not self.storage.persist_directory:
            raise ValueError(
                "storage.persist_directory cannot be empty."
            )

        valid_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if self.logging.level not in valid_levels:
            raise ValueError(
                f"Invalid logging level: {self.logging.level}. "
                f"Expected one of {sorted(valid_levels)}."
            )

        if self.logging.max_file_size_mb <= 0:
            raise ValueError(
                "logging.max_file_size_mb must be greater than 0."
            )

        if self.logging.backup_count < 0:
            raise ValueError(
                "logging.backup_count cannot be negative."
            )

    @property
    def persist_directory(self) -> Path:
        """Return expanded backend persistence directory."""

        return Path(
            self.storage.persist_directory
        ).expanduser()

    @property
    def log_directory(self) -> Path:
        """Return expanded logging directory."""

        return Path(
            self.logging.log_directory
        ).expanduser()

    @property
    def log_file(self) -> Path:
        """Return full path to the application log file."""

        return self.log_directory / self.logging.log_file

    def to_dict(self) -> dict:
        """Convert configuration to a serializable dictionary."""

        return {
            "knowledge_graph": {
                "enabled": self.knowledge_graph.enabled,
                "auto_build": self.knowledge_graph.auto_build,
            },

            "memory": {
                "enabled": self.memory.enabled,
                "session_tracking": self.memory.session_tracking,
            },

            "retrieval": {
                "enabled": self.retrieval.enabled,
                "top_k": self.retrieval.top_k,
            },

            "embeddings": {
                "provider": self.embeddings.provider,
                "model": self.embeddings.model,
            },

            "storage": {
                "persist_directory": self.storage.persist_directory,
            },

            "logging": {
                "level": self.logging.level,
                "log_directory": self.logging.log_directory,
                "log_file": self.logging.log_file,
                "console_enabled": self.logging.console_enabled,
                "file_enabled": self.logging.file_enabled,
                "max_file_size_mb": self.logging.max_file_size_mb,
                "backup_count": self.logging.backup_count,
            },
        }

    def save(self, filepath: Optional[str] = None) -> None:
        """Save configuration to YAML file."""

        if filepath is None:
            filepath = str(
                Path.home()
                / ".doc_explainer"
                / "config"
                / "backend_config.yaml"
            )

        path = Path(filepath).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
                indent=2,
            )

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""

        self.knowledge_graph = KnowledgeGraphConfig()
        self.memory = MemoryConfig()
        self.retrieval = RetrievalConfig()
        self.embeddings = EmbeddingConfig()
        self.storage = StorageConfig()
        self.logging = LoggingConfig()