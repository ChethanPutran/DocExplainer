from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import yaml
from dotenv import load_dotenv


load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for the LLM provider and inference."""

    provider: str = "openrouter"
    model: str = "moonshotai/kimi-k3"

    temperature: float = 1.0
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None

    requests_per_minute: int = 4
    min_request_interval_seconds: Optional[float] = None
    rate_limit_retries: int = 2

    reasoning: bool = True

    base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "LLMConfig":
        """Load LLM configuration from YAML."""

        if filepath is None:
            filepath = str(Path.home() / '.doc_explainer' / 'config' / 'llm_config.yaml')

        path = Path(filepath).expanduser() 

        if not path.exists():
            raise FileNotFoundError(
                f"LLM configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        llm_data = data.get("llm", {})

        return cls(
            provider=llm_data.get("provider", cls.provider),
            model=llm_data.get("model", cls.model),
            temperature=llm_data.get(
                "temperature",
                cls.temperature,
            ),
            max_tokens=llm_data.get("max_tokens"),
            timeout=llm_data.get("timeout"),
            requests_per_minute=llm_data.get(
                "requests_per_minute",
                cls.requests_per_minute,
            ),
            min_request_interval_seconds=llm_data.get(
                "min_request_interval_seconds"
            ),
            rate_limit_retries=llm_data.get(
                "rate_limit_retries",
                cls.rate_limit_retries,
            ),
            reasoning=llm_data.get(
                "reasoning",
                cls.reasoning,
            ),
            base_url=llm_data.get(
                "base_url",
                cls.base_url,
            ),
            ollama_base_url=llm_data.get(
                "ollama_base_url",
                cls.ollama_base_url,
            ),
        )

    @property
    def api_key(self) -> Optional[str]:
        """Return the API key for the configured provider."""

        provider = self.provider.lower()

        if provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY")

        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")

        if provider == "gemini":
            return os.getenv("GEMINI_API_KEY")

        if provider == "ollama":
            return None

        return None

    def validate(self) -> None:
        """Validate the current LLM configuration."""

        supported_providers = {
            "openrouter",
            "openai",
            "gemini",
            "ollama",
            "local",
        }

        if self.provider.lower() not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                f"Supported providers: {sorted(supported_providers)}"
            )

        if not self.model:
            raise ValueError("LLM model cannot be empty.")

        if not 0.0 <= self.temperature:
            raise ValueError(
                "LLM temperature must be >= 0."
            )

        if self.requests_per_minute <= 0:
            raise ValueError(
                "requests_per_minute must be greater than 0."
            )

        if self.rate_limit_retries < 0:
            raise ValueError(
                "rate_limit_retries cannot be negative."
            )

        if self.provider.lower() != "ollama":
            if not self.api_key:
                raise ValueError(
                    f"Missing API key for provider '{self.provider}'. "
                    f"Set the appropriate environment variable."
                )

    def to_dict(self) -> dict:
        """Convert configuration to a serializable dictionary."""

        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "requests_per_minute": self.requests_per_minute,
            "min_request_interval_seconds": (
                self.min_request_interval_seconds
            ),
            "rate_limit_retries": self.rate_limit_retries,
            "reasoning": self.reasoning,
            "base_url": self.base_url,
            "ollama_base_url": self.ollama_base_url,
        }

    def save(self, filepath: Optional[str] = None):
            """Save configuration to file"""
            if filepath is None:
                filepath = str(Path.home() / '.doc_explainer' / 'config' / 'llm_config.yaml')
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(self, f, indent=2)