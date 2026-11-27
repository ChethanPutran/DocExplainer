"""Layered configuration: ONE place for every default, overridable by file and environment.

    packaged defaults (config/defaults/<name>.yaml)      <- the only place a default value is written
      <- the user's file (~/.doc_explainer/config/<name>.yaml; may hold just the keys it changes)
      <- environment variables  DOC_EXPLAINER_<SECTION>_<KEY>   (e.g. DOC_EXPLAINER_LLM_PROVIDER=openai)

Secrets (API keys) are never in a file: they come from the environment / ``.env`` only.
"""
from __future__ import annotations

import copy
import logging
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import find_dotenv, load_dotenv

from .yaml_io import load_yaml

# Secrets live in a .env file (or the real environment): the one next to the project, else the one in the working directory.
load_dotenv()
load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)
ENV_PREFIX = "DOC_EXPLAINER"
CONFIG_DIR = Path.home() / ".doc_explainer" / "config"


@lru_cache(maxsize=None)
def _packaged(name: str) -> Dict[str, Any]:
    text = resources.files("doc_explainer.config").joinpath("defaults", f"{name}.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def packaged_defaults(name: str) -> Dict[str, Any]:
    """A private copy of the shipped defaults for ``name`` (``llm_config`` or ``backend_config``)."""
    return copy.deepcopy(_packaged(name))


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def env_overlay(data: Dict[str, Any], path: tuple = ()) -> Dict[str, Any]:
    """Apply ``DOC_EXPLAINER_<PATH>`` variables to every scalar setting. Values are parsed as YAML
    (``10`` -> 10, ``true`` -> True, ``null`` -> None)."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        here = path + (key,)
        if isinstance(value, dict):
            out[key] = env_overlay(value, here)
            continue
        raw = os.getenv("_".join((ENV_PREFIX,) + here).upper())
        if raw is None:
            out[key] = value
            continue
        try:
            out[key] = yaml.safe_load(raw)
        except yaml.YAMLError:
            out[key] = raw
        logger.info("Setting %s taken from the environment", ".".join(here))
    return out


def load_layered(name: str, user_file: Optional[str] = None) -> Dict[str, Any]:
    """The effective settings for ``name``. A missing user file is fine (the defaults apply)."""
    data = packaged_defaults(name)
    path = Path(user_file).expanduser() if user_file else CONFIG_DIR / f"{name}.yaml"
    if path.exists():
        data = deep_merge(data, load_yaml(path))
    elif user_file:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return env_overlay(data)


def known_keys_only(cls, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The entries of ``data`` that ``cls`` has fields for (keys from older files are dropped, with a note)."""
    known = cls.__dataclass_fields__
    unknown = sorted(set(data or {}) - set(known))
    if unknown:
        logger.warning("Ignoring unknown %s setting(s): %s", cls.__name__, ", ".join(unknown))
    return {k: v for k, v in (data or {}).items() if k in known}


def difference(current: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Only what differs from ``base`` (what is worth writing to a user file)."""
    out: Dict[str, Any] = {}
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            sub = difference(value, base[key])
            if sub:
                out[key] = sub
        elif base.get(key) != value:
            out[key] = value
    return out
