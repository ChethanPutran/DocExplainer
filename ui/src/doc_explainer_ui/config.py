"""UI settings. Every default lives in ``defaults/ui_config.yaml``; nothing is repeated here.

    packaged defaults  <-  ~/.doc_explainer/config/ui_config.yaml (only the keys the user changed)
                       <-  environment DOC_EXPLAINER_UI_<SECTION>_<KEY>   (e.g. DOC_EXPLAINER_UI_SERVER_URL)

The language model and the knowledge graph are the server's settings, not the UI's.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".doc_explainer" / "config"
CONFIG_FILE = CONFIG_DIR / "ui_config.yaml"
ENV_PREFIX = "DOC_EXPLAINER_UI"


def packaged_defaults() -> Dict[str, Any]:
    text = resources.files("doc_explainer_ui").joinpath("defaults", "ui_config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _env(data: Dict[str, Any], path: tuple = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in data.items():
        here = path + (k,)
        if isinstance(v, dict):
            out[k] = _env(v, here)
            continue
        raw = os.getenv("_".join((ENV_PREFIX,) + here).upper())
        try:
            out[k] = v if raw is None else yaml.safe_load(raw)
        except yaml.YAMLError:
            out[k] = raw
    return out


def _difference(cur: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in cur.items():
        if isinstance(v, dict):
            sub = _difference(v, base.get(k, {}))
            if sub:
                out[k] = sub
        elif base.get(k) != v:
            out[k] = v
    return out


def _pick(cls, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    unknown = sorted(set(data or {}) - set(cls.__dataclass_fields__))
    if unknown:
        logger.warning("Ignoring unknown %s setting(s): %s", cls.__name__, ", ".join(unknown))
    return {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}


_D = packaged_defaults()


@dataclass
class WindowConfig:
    width: int = _D["window"]["width"]
    height: int = _D["window"]["height"]
    title: str = _D["window"]["title"]
    maximized: bool = _D["window"]["maximized"]


@dataclass
class ThemeConfig:
    name: str = _D["theme"]["name"]
    mode: str = _D["theme"]["mode"]
    font_size: int = _D["theme"]["font_size"]
    font_family: str = _D["theme"]["font_family"]


@dataclass
class SidebarConfig:
    width: int = _D["sidebar"]["width"]
    visible: bool = _D["sidebar"]["visible"]
    position: str = _D["sidebar"]["position"]


@dataclass
class VoiceConfig:
    enabled: bool = _D["voice"]["enabled"]
    input_device: str = _D["voice"]["input_device"]
    whisper_model: str = _D["voice"]["whisper_model"]
    model_idle_unload_s: float = _D["voice"]["model_idle_unload_s"]
    preload_model: bool = _D["voice"]["preload_model"]
    output_enabled: bool = _D["voice"]["output_enabled"]
    output_rate: int = _D["voice"]["output_rate"]
    output_volume: float = _D["voice"]["output_volume"]
    output_engine: str = _D["voice"]["output_engine"]
    piper_voice: str = _D["voice"]["piper_voice"]
    piper_voices_dir: str = _D["voice"]["piper_voices_dir"]


@dataclass
class DocumentConfig:
    default_zoom: float = _D["documents"]["default_zoom"]
    max_recent_files: int = _D["documents"]["max_recent_files"]
    auto_save_interval: int = _D["documents"]["auto_save_interval"]
    default_path: str = _D["documents"]["default_path"]


@dataclass
class CacheConfig:
    enabled: bool = _D["cache"]["enabled"]
    size_mb: int = _D["cache"]["size_mb"]
    location: str = _D["cache"]["location"]


@dataclass
class StartupConfig:
    open_last_docs: bool = _D["startup"]["open_last_docs"]
    check_updates: bool = _D["startup"]["check_updates"]


@dataclass
class KnowledgeConfig:
    build_graph: str = _D["knowledge"]["build_graph"]          # always | ask | never


@dataclass
class SystemConfig:
    debug_mode: bool = _D["system"]["debug_mode"]
    log_level: str = _D["system"]["log_level"]
    show_splash: bool = _D["system"]["show_splash"]
    enable_profiling: bool = _D["system"]["enable_profiling"]
    send_usage_stats: bool = _D["system"]["send_usage_stats"]
    allow_telemetry: bool = _D["system"]["allow_telemetry"]


@dataclass
class ServerConfig:
    """Where the backend service is. The API key is deliberately not stored in a file (DOC_EXPLAINER_API_KEY)."""
    url: str = _D["server"]["url"]
    protocol: str = _D["server"]["protocol"]
    timeout_s: float = _D["server"]["timeout_s"]
    user_id: str = _D["server"]["user_id"]


_SECTIONS = {"window": WindowConfig, "theme": ThemeConfig, "sidebar": SidebarConfig, "voice": VoiceConfig,
             "documents": DocumentConfig, "cache": CacheConfig, "startup": StartupConfig, "knowledge": KnowledgeConfig, "system": SystemConfig,
             "server": ServerConfig}


@dataclass
class UIConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    sidebar: SidebarConfig = field(default_factory=SidebarConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "UIConfig":
        """defaults <- user file <- environment. A missing default-location file is fine; a missing explicit one is not."""
        path = Path(filepath).expanduser() if filepath else CONFIG_FILE
        data = packaged_defaults()
        if path.exists():
            try:
                data = _merge(data, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            except yaml.YAMLError as e:
                raise ValueError(
                    f"{path} is not readable YAML (older versions wrote Python-object tags that cannot be loaded "
                    f"safely). Move it aside and the defaults apply.\nDetails: {str(e).splitlines()[0]}") from e
        elif filepath:
            raise FileNotFoundError(f"UI configuration file not found: {path}")
        data = _env(data)
        return cls(**{name: klass(**_pick(klass, data.get(name))) for name, klass in _SECTIONS.items()})

    def save(self, filepath: Optional[str] = None) -> None:
        """Write only what differs from the defaults, so a later change of a default still reaches this user."""
        path = Path(filepath).expanduser() if filepath else CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_difference(asdict(self), packaged_defaults()), sort_keys=False), encoding="utf-8")
