"""Server settings. Everything is configurable by env var (``DOC_EXPLAINER_*``) or CLI flag."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    # Empty = no authentication. The CLI refuses that on a non-loopback host.
    api_key: str = ""
    default_user_id: str = "default"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".doc_explainer" / "server")
    max_upload_mb: int = 50
    max_queued_jobs: int = 20        # waiting + running, whole server
    max_jobs_per_user: int = 5       # waiting + running, per user
    enable_rest: bool = True
    enable_rpc: bool = True
    llm_config_path: Optional[str] = None
    backend_config_path: Optional[str] = None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def protocols(self) -> List[str]:
        return [p for p, on in (("rest", self.enable_rest), ("rpc", self.enable_rpc)) if on]

    @classmethod
    def from_env(cls) -> "ApiSettings":
        d = cls()
        return cls(
            host=os.getenv("DOC_EXPLAINER_HOST", d.host),
            port=int(os.getenv("DOC_EXPLAINER_PORT", d.port)),
            api_key=os.getenv("DOC_EXPLAINER_API_KEY", d.api_key),
            default_user_id=os.getenv("DOC_EXPLAINER_USER_ID", d.default_user_id),
            data_dir=Path(os.getenv("DOC_EXPLAINER_DATA_DIR", str(d.data_dir))).expanduser(),
            max_upload_mb=int(os.getenv("DOC_EXPLAINER_MAX_UPLOAD_MB", d.max_upload_mb)),
            max_queued_jobs=int(os.getenv("DOC_EXPLAINER_MAX_QUEUED_JOBS", d.max_queued_jobs)),
            max_jobs_per_user=int(os.getenv("DOC_EXPLAINER_MAX_JOBS_PER_USER", d.max_jobs_per_user)),
            enable_rest=_flag("DOC_EXPLAINER_ENABLE_REST", True),
            enable_rpc=_flag("DOC_EXPLAINER_ENABLE_RPC", True),
            llm_config_path=os.getenv("DOC_EXPLAINER_LLM_CONFIG") or None,
            backend_config_path=os.getenv("DOC_EXPLAINER_BACKEND_CONFIG") or None,
        )
