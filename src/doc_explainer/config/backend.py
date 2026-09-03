from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional

import yaml


# ================================================================
# NEO4J
# ================================================================

@dataclass
class Neo4jConfig:
    """Neo4j graph database configuration."""

    uri: str = field(
        default_factory=lambda: os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )
    )

    user: str = field(
        default_factory=lambda: os.getenv(
            "NEO4J_USER",
            "neo4j",
        )
    )

    password: str = field(
        default_factory=lambda: os.getenv(
            "NEO4J_PASSWORD",
            "",
        )
    )


# ================================================================
# KNOWLEDGE GRAPH
# ================================================================

@dataclass
class KnowledgeGraphConfig:
    """Knowledge graph configuration."""

    enabled: bool = True
    auto_build: bool = True


# ================================================================
# MEMORY
# ================================================================

@dataclass
class MemoryConfig:
    """Memory and session tracking configuration."""

    enabled: bool = True
    session_tracking: bool = True


# ================================================================
# RETRIEVAL
# ================================================================

@dataclass
class RetrievalConfig:
    """Document retrieval configuration."""

    enabled: bool = True
    top_k: int = 5


# ================================================================
# EMBEDDINGS
# ================================================================

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    provider: str = "default"
    model: str = ""


# ================================================================
# STORAGE
# ================================================================

@dataclass
class StorageConfig:
    """Backend storage configuration."""

    persist_directory: str = "~/.doc_explainer/cache"


# ================================================================
# LOGGING
# ================================================================

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

class VectorStoreConfig:
    """Vector store configuration."""

    persist_directory: str = "~/.doc_explainer/vector_store"

# ================================================================
# BACKEND CONFIGURATION
# ================================================================

@dataclass
class BackendConfig:
    """Configuration for backend application services."""

    knowledge_graph: KnowledgeGraphConfig = field(
        default_factory=KnowledgeGraphConfig
    )

    neo4j: Neo4jConfig = field(
        default_factory=Neo4jConfig
    )

    vector_store: VectorStoreConfig = field(
        default_factory=VectorStoreConfig
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

    # ============================================================
    # PATH
    # ============================================================

    @classmethod
    def default_path(cls) -> Path:
        """Return the default backend configuration path."""

        return (
            Path.home()
            / ".doc_explainer"
            / "config"
            / "backend_config.yaml"
        )

    # ============================================================
    # LOAD
    # ============================================================

    @classmethod
    def load(
        cls,
        filepath: Optional[str] = None,
        create_if_missing: bool = True,
    ) -> "BackendConfig":
        """
        Load backend configuration from YAML.

        Environment variables take precedence over YAML values
        for Neo4j connection settings.

        Supported environment variables:

            NEO4J_URI
            NEO4J_USER
            NEO4J_PASSWORD

        Args:
            filepath:
                Path to the configuration file.

            create_if_missing:
                Whether to create the default configuration file
                when it does not exist.

        Returns:
            Loaded BackendConfig.

        Raises:
            FileNotFoundError:
                If the configuration file does not exist and
                create_if_missing is False.

            yaml.YAMLError:
                If the YAML file is malformed.

            ValueError:
                If the YAML structure is invalid.
        """

        path = (
            Path(filepath).expanduser()
            if filepath is not None
            else cls.default_path()
        )

        # ---------------------------------------------------------
        # Configuration does not exist
        # ---------------------------------------------------------

        if not path.exists():

            config = cls()

            if create_if_missing:
                config.save(filepath=str(path))

            return config

        # ---------------------------------------------------------
        # Read YAML
        # ---------------------------------------------------------

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"Backend configuration must contain "
                f"a YAML mapping: {path}"
            )

        # ---------------------------------------------------------
        # Extract configuration sections
        # ---------------------------------------------------------

        kg_data = data.get("knowledge_graph", {})
        neo4j_data = data.get("neo4j", {})
        memory_data = data.get("memory", {})
        retrieval_data = data.get("retrieval", {})
        embedding_data = data.get("embeddings", {})
        storage_data = data.get("storage", {})
        logging_data = data.get("logging", {})

        # ---------------------------------------------------------
        # Build configuration
        # ---------------------------------------------------------

        config = cls(

            # -----------------------------------------------------
            # Knowledge graph
            # -----------------------------------------------------

            knowledge_graph=KnowledgeGraphConfig(
                enabled=kg_data.get(
                    "enabled",
                    True,
                ),
                auto_build=kg_data.get(
                    "auto_build",
                    True,
                ),
            ),

            # -----------------------------------------------------
            # Neo4j
            #
            # Environment variables take precedence over YAML.
            # -----------------------------------------------------

            neo4j=Neo4jConfig(
                uri=os.getenv(
                    "NEO4J_URI",
                    neo4j_data.get(
                        "uri",
                        "bolt://localhost:7687",
                    ),
                ),

                user=os.getenv(
                    "NEO4J_USER",
                    neo4j_data.get(
                        "user",
                        "neo4j",
                    ),
                ),

                password=os.getenv(
                    "NEO4J_PASSWORD",
                    "",
                ),
            ),

            # -----------------------------------------------------
            # Memory
            # -----------------------------------------------------

            memory=MemoryConfig(
                enabled=memory_data.get(
                    "enabled",
                    True,
                ),
                session_tracking=memory_data.get(
                    "session_tracking",
                    True,
                ),
            ),

            # -----------------------------------------------------
            # Retrieval
            # -----------------------------------------------------

            retrieval=RetrievalConfig(
                enabled=retrieval_data.get(
                    "enabled",
                    True,
                ),
                top_k=retrieval_data.get(
                    "top_k",
                    5,
                ),
            ),

            # -----------------------------------------------------
            # Embeddings
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # Storage
            # -----------------------------------------------------

            storage=StorageConfig(
                persist_directory=storage_data.get(
                    "persist_directory",
                    "~/.doc_explainer/cache",
                ),
            ),

            # -----------------------------------------------------
            # Logging
            # -----------------------------------------------------

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

        return config

    # ================================================================
    # VALIDATION
    # ================================================================

    def validate(self) -> None:
        """Validate backend configuration."""

        # ------------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------------

        if self.retrieval.top_k <= 0:
            raise ValueError(
                "retrieval.top_k must be greater than 0."
            )

        # ------------------------------------------------------------
        # Storage
        # ------------------------------------------------------------

        if not self.storage.persist_directory:
            raise ValueError(
                "storage.persist_directory cannot be empty."
            )

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Neo4j
        # ------------------------------------------------------------

        if not self.neo4j.uri:
            raise ValueError(
                "neo4j.uri cannot be empty."
            )

        if not self.neo4j.user:
            raise ValueError(
                "neo4j.user cannot be empty."
            )

        if not self.neo4j.password:
            raise ValueError(
                "NEO4J_PASSWORD must be set."
            )

    # ================================================================
    # PROPERTIES
    # ================================================================

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

        return (
            self.log_directory
            / self.logging.log_file
        )

    # ================================================================
    # SERIALIZATION
    # ================================================================

    def to_dict(self) -> dict:
        """
        Convert configuration to a serializable dictionary.

        Neo4j password is intentionally excluded.
        """

        return {

            # --------------------------------------------------------
            # Knowledge graph
            # --------------------------------------------------------

            "knowledge_graph": {
                "enabled": self.knowledge_graph.enabled,
                "auto_build": self.knowledge_graph.auto_build,
            },

            # --------------------------------------------------------
            # Neo4j
            #
            # Password is intentionally NOT persisted.
            # --------------------------------------------------------

            "neo4j": {
                "uri": self.neo4j.uri,
                "user": self.neo4j.user,
            },

            # --------------------------------------------------------
            # Memory
            # --------------------------------------------------------

            "memory": {
                "enabled": self.memory.enabled,
                "session_tracking": self.memory.session_tracking,
            },

            # --------------------------------------------------------
            # Retrieval
            # --------------------------------------------------------

            "retrieval": {
                "enabled": self.retrieval.enabled,
                "top_k": self.retrieval.top_k,
            },

            # --------------------------------------------------------
            # Embeddings
            # --------------------------------------------------------

            "embeddings": {
                "provider": self.embeddings.provider,
                "model": self.embeddings.model,
            },

            # --------------------------------------------------------
            # Storage
            # --------------------------------------------------------

            "storage": {
                "persist_directory": str(
                    self.storage.persist_directory
                ),
            },

            # --------------------------------------------------------
            # Logging
            # --------------------------------------------------------

            "logging": {
                "level": self.logging.level,
                "log_directory": str(
                    self.logging.log_directory
                ),
                "log_file": self.logging.log_file,
                "console_enabled": self.logging.console_enabled,
                "file_enabled": self.logging.file_enabled,
                "max_file_size_mb": self.logging.max_file_size_mb,
                "backup_count": self.logging.backup_count,
            },
        }

    # ================================================================
    # SAVE
    # ================================================================

    def save(
        self,
        filepath: Optional[str] = None,
    ) -> None:
        """Save configuration to YAML file."""

        path = (
            Path(filepath).expanduser()
            if filepath is not None
            else self.default_path()
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
                indent=2,
            )

    # ================================================================
    # RESET
    # ================================================================

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""

        self.knowledge_graph = (
            KnowledgeGraphConfig()
        )

        self.neo4j = (
            Neo4jConfig()
        )

        self.memory = (
            MemoryConfig()
        )

        self.retrieval = (
            RetrievalConfig()
        )

        self.embeddings = (
            EmbeddingConfig()
        )

        self.storage = (
            StorageConfig()
        )

        self.logging = (
            LoggingConfig()
        )