from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class WindowConfig:
    width: int = 1200
    height: int = 800
    title: str = "Doc Explainer"
    maximized: bool = False


@dataclass
class ThemeConfig:
    name: str = "test"
    mode: str = "light"
    font_size: int = 10
    font_family: str = "Segoe UI"


@dataclass
class SidebarConfig:
    width: int = 350
    visible: bool = True
    position: str = "right"


@dataclass
class VoiceConfig:
    enabled: bool = True
    input_device: str = "default"
    output_enabled: bool = True
    output_rate: int = 150
    output_volume: float = 0.9


@dataclass
class DocumentConfig:
    default_zoom: float = 1.0
    max_recent_files: int = 10
    auto_save_interval: int = 5
    default_path: str = "~/Documents"


@dataclass
class CacheConfig:
    enabled: bool = True
    size_mb: int = 500
    location: str = "~/.doc_explainer/cache"


@dataclass
class StartupConfig:
    open_last_docs: bool = True
    check_updates: bool = True


@dataclass
class SystemConfig:
    debug_mode: bool = False
    log_level: str = "INFO"
    show_splash: bool = True
    enable_profiling: bool = False
    send_usage_stats: bool = False
    allow_telemetry: bool = False


@dataclass
class UIConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    sidebar: SidebarConfig = field(default_factory=SidebarConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def default_path(cls) -> Path:
        return (
            Path.home()
            / ".doc_explainer"
            / "config"
            / "ui_config.yaml"
        )

    @classmethod
    def load(
        cls,
        filepath: Optional[str] = None,
    ) -> "UIConfig":

        path = (
            Path(filepath).expanduser()
            if filepath is not None
            else cls.default_path()
        )

        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"UI configuration must contain a YAML mapping: {path}"
            )

        return cls(
            window=WindowConfig(
                **data.get("window", {})
            ),

            theme=ThemeConfig(
                **data.get("theme", {})
            ),

            sidebar=SidebarConfig(
                **data.get("sidebar", {})
            ),

            voice=VoiceConfig(
                **data.get("voice", {})
            ),

            documents=DocumentConfig(
                **data.get("documents", {})
            ),

            cache=CacheConfig(
                **data.get("cache", {})
            ),

            startup=StartupConfig(
                **data.get("startup", {})
            ),

            system=SystemConfig(
                **data.get("system", {})
            ),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a plain serializable dictionary."""

        return asdict(self)

    def save(
        self,
        filepath: Optional[str] = None,
    ) -> None:

        path = (
            Path(filepath).expanduser()
            if filepath is not None
            else self.default_path()
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
                indent=2,
                default_flow_style=False,
            )