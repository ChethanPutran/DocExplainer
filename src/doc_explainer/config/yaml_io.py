"""YAML reading for config files, with errors a user can act on."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigFileError(ValueError):
    """A config file exists but cannot be read."""


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigFileError(
            f"{path} is not readable YAML (older versions wrote Python-object tags that cannot be "
            f"loaded safely). Move it aside and the defaults will be re-created on next start.\n"
            f"Details: {str(e).splitlines()[0]}") from e
    if not isinstance(data, dict):
        raise ConfigFileError(f"{path} must contain a YAML mapping at the top level.")
    return data
