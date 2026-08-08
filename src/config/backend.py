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

    @classmethod
    def load(cls,  filepath: Optional[str] = None) -> "BackendConfig":
        """Load backend configuration from YAML."""

        if filepath is None:
            filepath = str(Path.home() / '.doc_explainer' / 'config' / 'backend_config.yaml')

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

    @property
    def persist_directory(self) -> Path:
        """Return expanded backend persistence directory."""

        return Path(
            self.storage.persist_directory
        ).expanduser()

    def to_dict(self) -> dict:
        """Convert configuration to a serializable dictionary."""

        return {
            "knowledge_graph": {
                "enabled": self.knowledge_graph.enabled,
                "auto_build": self.knowledge_graph.auto_build,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "session_tracking": (
                    self.memory.session_tracking
                ),
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
                "persist_directory": (
                    self.storage.persist_directory
                ),
            },
        }

    
    def save(self, filepath: Optional[str] = None):
            """Save configuration to file"""
            if filepath is None:
                filepath = str(Path.home() / '.doc_explainer' / 'config' / 'backend_config.yaml')
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(self, f, indent=2)

    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.knowledge_graph = KnowledgeGraphConfig()
        self.memory = MemoryConfig()
        self.retrieval = RetrievalConfig()
        self.embeddings = EmbeddingConfig()
        self.storage = StorageConfig()
        