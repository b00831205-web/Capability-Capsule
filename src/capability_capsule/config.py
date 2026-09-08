"""Typed project configuration for the Capability Capsule skeleton."""

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    """Base configuration model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class OllamaConfig(StrictModel):
    """Local Ollama endpoint and model selections."""

    base_url: HttpUrl = HttpUrl("http://127.0.0.1:11434")
    generation_model: str = "qwen3.5:4b"
    embedding_model: str = "nomic-embed-text"


class HttpConfig(StrictModel):
    """HTTP environment behavior for local service calls."""

    trust_env: bool = False


class CapsuleBudgetConfig(StrictModel):
    """User-provided resource and offline-duration constraints."""

    size_budget_bytes: int = Field(default=1_073_741_824, gt=0)
    offline_duration_hours: float = Field(default=12.0, gt=0)


class TelemetryConfig(StrictModel):
    """Telemetry enablement and local output location."""

    enabled: bool = True
    output_dir: Path = Path(".capsule/sessions")


class RagConfig(StrictModel):
    """Limits for retrieved source text supplied to generation"""

    max_context_chars: int = Field(default=12_000, gt=0)


class Settings(StrictModel):
    """Top-level Capability Capsule settings."""

    ollama: OllamaConfig = OllamaConfig()
    http: HttpConfig = HttpConfig()
    capsule: CapsuleBudgetConfig = CapsuleBudgetConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    rag: RagConfig = RagConfig()

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        """Load and validate a TOML configuration file."""

        with path.open("rb") as stream:
            data: dict[str, Any] = tomllib.load(stream)
        return cls.model_validate(data)
