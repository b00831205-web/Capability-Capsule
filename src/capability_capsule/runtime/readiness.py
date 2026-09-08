"""Check whether a capsule is ready o run with local Ollama"""

from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict

from capability_capsule.packager.capsule import inspect_capsule


class ReadinessReport(BaseModel):
    """Result of validating a capsule and its required local models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capsule_path: Path
    capsule_build_id: UUID
    ollama_version: str
    required_models: tuple[str, ...]
    installed_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    ready: bool


def _model_is_available(required: str, installed: set[str]) -> bool:
    """Match exact model names and Ollama's implicit latest tag."""

    if required in installed:
        return True

    return ":" not in required and f"{required}:latest" in installed


def _read_version(payload: Any) -> str:
    """Read and validate on Ollama version response"""

    if not isinstance(payload, dict):
        raise ValueError("Ollama version response must be an object")

    version = payload.get("version")

    if not isinstance(version, str) or not version.strip():
        raise ValueError("Ollama version response is missing a version")

    return version


def _read_model_names(payload: Any) -> set[str]:
    """Read model names from an Ollama tags response"""

    if not isinstance(payload, dict):
        raise ValueError("Ollama tags response must be an object")

    models = payload.get("models")

    if not isinstance(models, list):
        raise ValueError("Ollama tags response is missing its model list")

    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each Ollama model entry must be an object")

        for field in ("name", "model"):
            value = model.get(field)

            if isinstance(value, str) and value.strip():
                names.add(value)

    return names


def check_capsule_readiness(
    capsule_path: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ReadinessReport:
    """Validate a capsule and check that Ollama has its required models"""

    inspection = inspect_capsule(capsule_path)
    settings = inspection.settings

    with httpx.Client(
        base_url=str(settings.ollama.base_url),
        trust_env=settings.http.trust_env,
        timeout=10.0,
        transport=transport,
    ) as client:
        version_response = client.get("/api/version")
        version_response.raise_for_status()

        tags_response = client.get("/api/tags")
        tags_response.raise_for_status()

    version = _read_version(version_response.json())
    installed = _read_model_names(tags_response.json())

    required = tuple(
        dict.fromkeys(
            (
                settings.ollama.generation_model,
                settings.ollama.embedding_model,
            )
        )
    )
    missing = tuple(model for model in required if not _model_is_available(model, installed))

    return ReadinessReport(
        capsule_path=inspection.path,
        capsule_build_id=inspection.manifest.capsule_build_id,
        ollama_version=version,
        required_models=required,
        installed_models=tuple(sorted(installed)),
        missing_models=missing,
        ready=not missing,
    )
