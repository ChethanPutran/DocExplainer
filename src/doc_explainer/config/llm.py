"""LLM settings. Every default lives in ``defaults/llm_config.yaml`` (see ``layered``); nothing is repeated here."""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .layered import CONFIG_DIR, difference, known_keys_only, load_layered, packaged_defaults

_D = packaged_defaults("llm_config")["llm"]
_PROVIDERS = lambda: copy.deepcopy(packaged_defaults("llm_config")["providers"])  # noqa: E731


@dataclass
class LLMConfig:
    provider: str = _D["provider"]
    model: Optional[str] = _D["model"]                     # None = the provider's default
    base_url: Optional[str] = _D["base_url"]               # None = the provider's endpoint
    temperature: float = _D["temperature"]
    max_tokens: Optional[int] = _D["max_tokens"]
    max_tokens_cap: Optional[int] = _D["max_tokens_cap"]
    token_growth: float = _D["token_growth"]
    token_retries: int = _D["token_retries"]
    timeout: Optional[int] = _D["timeout"]
    requests_per_minute: int = _D["requests_per_minute"]
    min_request_interval_seconds: Optional[float] = _D["min_request_interval_seconds"]
    rate_limit_retries: int = _D["rate_limit_retries"]
    reasoning: bool = _D["reasoning"]
    ollama_base_url: str = _D["ollama_base_url"]
    providers: Dict[str, Dict[str, Any]] = field(default_factory=_PROVIDERS)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "LLMConfig":
        """defaults <- user file (``filepath`` or ``~/.doc_explainer/config/llm_config.yaml``) <- environment."""
        data = load_layered("llm_config", filepath)
        return cls(**known_keys_only(cls, {**data.get("llm", {}), "providers": data.get("providers", {})}))

    # ------------------------------------------------------------------ what applies to the chosen provider
    @property
    def profile(self) -> Dict[str, Any]:
        return self.providers.get(self.provider.lower(), {})

    @property
    def api_key(self) -> Optional[str]:
        env = self.profile.get("api_key_env")
        return os.getenv(env) if env else None

    @property
    def resolved_model(self) -> str:
        return self.model or self.profile.get("model") or ""

    @property
    def resolved_base_url(self) -> Optional[str]:
        return self.base_url or self.profile.get("base_url")

    def client_kwargs(self) -> dict:
        """Extra arguments a provider's wrapper needs to reach its endpoint (the others read their key from the environment)."""
        provider = self.provider.lower()
        if provider == "airouter":
            return {"api_key": self.api_key, "base_url": self.resolved_base_url}
        if provider == "ollama":
            return {"base_url": self.ollama_base_url}
        return {}

    def validate(self) -> None:
        if self.provider.lower() not in self.providers:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Supported providers: {sorted(self.providers)}")
        if not self.resolved_model:
            raise ValueError(f"No model for provider '{self.provider}': set llm.model or providers.{self.provider}.model.")
        if self.temperature < 0:
            raise ValueError("LLM temperature must be >= 0.")
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0.")
        if self.rate_limit_retries < 0:
            raise ValueError("rate_limit_retries cannot be negative.")
        env = self.profile.get("api_key_env")
        if env and not self.api_key:
            raise ValueError(f"Missing API key for provider '{self.provider}': set {env} (environment or .env).")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, filepath: Optional[str] = None) -> None:
        """Write only what differs from the defaults, so a later change of a default still reaches this user."""
        path = Path(filepath).expanduser() if filepath else CONFIG_DIR / "llm_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        base = packaged_defaults("llm_config")
        current = self.to_dict()
        out = {"llm": difference({k: v for k, v in current.items() if k != "providers"}, base["llm"])}
        providers = difference(current["providers"], base["providers"])
        if providers:
            out["providers"] = providers
        path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
