"""Immutable, digest-verified coding-harness profiles."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HarnessToolProfile(BaseModel):
    """One Student-visible tool contract exposed by a harness"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments_schema: dict[str, Any]
    result_envelope: str = Field(min_length=1)

    @field_validator("name", "result_envelope")
    @classmethod
    def reject_blank_text(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("Harness tool fields must noe be blank")

        return value

    @field_validator("arguments_schema")
    @classmethod
    def validate_arguments_schema(cls, value: dict[str, Any])-> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError(
                "Harness tool arguments_shcema must describe an object"
            )

        properties = value.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(
                "Harness tool arguments_schema must contain properties"
            )

        required = value.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required) or len(set(required))!=len(required):
            raise ValueError(
                "Harness tool required arguments must be unique strings"
            )

        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise ValueError(
                "Harness tool required arguments must exist in properties"
            )

        if value.get("additionalProperties") is not False:
            raise ValueError(
                "Harness tool arguments_schema must reject additional properties"
            )

        return value

class HarnessProfile(BaseModel):
    """Pinned behavior required to align data with one coding harness"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    profile_id: str = Field(min_length=1)
    harness_id: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    provider_protocol: str = Field(min_length=1)
    model_injection: str = Field(min_length=1)
    prompt_template_sha256: str = Field(pattern =r"[0-9a-f]{64}$")
    fixed_context_tokens: int = Field(ge=0)
    shell: str = Field(min_length=1)
    filesystem: str = Field(min_length=1)
    approval_policy: str = Field(min_length=1)
    sandbox_policy: str = Field(min_length=1)
    supports_parallel_tools: bool
    supports_multi_turn_tools: bool
    tools: tuple[HarnessToolProfile, ...] = Field(min_length=1)
    validator_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "profile_id",
        "harness_id",
        "harness_version",
        "provider_protocol",
        "model_injection",
        "shell",
        "filesystem",
        "approval_policy",
        "sandbox_policy",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harness profile fields must not be blank")

        return value

    @field_validator("validator_ids")
    @classmethod
    def validate_validator_ids(
        cls,
        value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("Harness validator IDs must not be blank")

        if len(set(value)) != len(value):
            raise ValueError("Harness validator IDs must be unique")

        return value

    @model_validator(mode="after")
    def validate_unique_tools(self) -> Self:
        tool_names = tuple(tool.name for tool in self.tools)

        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Harness tool names must be unique")

        return self

class HarnessProfileReference(BaseModel):
    """Immutable reference stored in a collection plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str =Field(min_length=1)
    harness_id: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "profile_id",
        "harness_id",
        "harness_version",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Harness profile reference fields must not be blank")

        return value

    @field_validator("path")
    @classmethod
    def require_json_path(cls, value: Path) -> Path:
        if value.suffix.lower() != ".json":
            raise ValueError(
                "Harness profile path must use the .json extension"
            )

        return value

def _resolve_artifact(path: Path, artifact_root: Path | None) -> Path:
    if path.is_absolute():
        return path

    return (artifact_root or Path.cwd()) /path

def verify_harness_profile(reference: HarnessProfileReference, *, artifact_root: Path | None = None) -> HarnessProfile:
    """Load and verify the exact harness profile referenced by a plan"""

    profile_path = _resolve_artifact(reference.path, artifact_root)

    try:
        payload = profile_path.read_bytes()

    except OSError as error:
        raise ValueError(
            f"Harness profile cannot be read: {profile_path}"
        ) from error

    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != reference.sha256:
        raise ValueError(
            f"Harness profile SHA-256 mismatch for {profile_path}: "
            f"expected {reference.sha256}, got {actual_sha256}"
        )

    try:
        raw_profile = json.loads(payload)

    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Harness profile is not valid JSON: {profile_path}"
        ) from error

    if not isinstance(raw_profile, dict):
        raise ValueError(
            f"Harness profile must contain a JSON object: {profile_path}"
        )

    profile = HarnessProfile.model_validate(raw_profile)

    expected_identity = {
        "profile_id": reference.profile_id,
        "harness_id": reference.harness_id,
        "harness_version": reference.harness_version,
    }

    for field_name, expected_value in expected_identity.items():
        actual_value = getattr(profile, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"Harness profile field {field_name!r} does not match the collection-plan reference: expected {expected_value!r}, got {actual_value!r}"
            )
    return profile