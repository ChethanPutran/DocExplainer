"""First-run setup of the user's config directory.

The defaults ship inside the package and are read from there on every start; they are NOT copied, so a
better default in a new release reaches everyone. What is created here is a small *override* file per
config, holding the shipped defaults as comments to edit. A pip/uv install cannot write to the user's
home directory (wheels run no post-install code), so this happens when the server first starts or on
demand with ``doc-explainer-server --init-config``. Existing files are never overwritten.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from importlib import resources

from .layered import CONFIG_DIR

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = CONFIG_DIR
DEFAULT_FILES = ("llm_config.yaml", "backend_config.yaml")

_HEADER = ("# Your overrides. Anything you leave out uses the packaged default; settings can also come from\n"
           "# DOC_EXPLAINER_<SECTION>_<KEY> environment variables. Uncomment a line to change it.\n"
           "# The defaults currently are:\n#\n")


def _stub(name: str) -> str:
    text = resources.files("doc_explainer.config").joinpath("defaults", name).read_text(encoding="utf-8")
    return _HEADER + "".join(f"# {line}" if line.strip() else "#\n" for line in text.splitlines(keepends=True))


def ensure_default_configs(config_dir: Optional[Path] = None) -> List[Path]:
    """Create any missing override file in ``config_dir``; return the files created."""
    target = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    created: List[Path] = []
    target.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_FILES:
        dest = target / name
        if dest.exists():
            continue
        dest.write_text(_stub(name), encoding="utf-8")
        created.append(dest)
        logger.info("Created config file %s", dest)
    return created
