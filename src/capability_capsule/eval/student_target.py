"""Pinned Student deployment inputs for Teacher collection."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EnvironmentPurpose(StrEnum):
    """A purpose served by one observed hardware environment."""

    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    QUANTIZATION = "quantization"
    BENCHMARKING = "benchmarking"
    DATA_GENERATION = "data_generation"

class StudentModelRole(StrEnum):
    """A responsibility assigned to a Student model."""

    ROUTER = "router"
    EXECUTOR = "executor"
    CODER = "coder"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    REVIEWER = "reviewer"
    SUMMARIZER = "summarizer"

class HardwareProfileReference(BaseModel):
    """Immutable reference to a verified Student deployment profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_status: Literal["confirmed"]
    purposes: tuple[EnvironmentPurpose, ...] = Field(
        min_length=1
    )

    @field_validator("profile_id")
    @classmethod
    def reject_blank_profile_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile_id must not be blank")
        return value

    @field_validator("path")
    @classmethod
    def require_json_path(cls, value: Path) -> Path:
        if value.suffix.lower() != ".json":
            raise ValueError("Hardware profile path must use the .json extension")
        return value

    @field_validator("purposes")
    @classmethod
    def require_unique_purpose(cls, value: tuple[EnvironmentPurpose, ...]):
        if len(set(value)) != len(value):
            raise ValueError(
                "Hardware environment purposes must be unique"
            )
        return value


class BaseModelRecommendationReference(BaseModel):
    """Accepted base-model recommendation pinned for collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation_id: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_status: Literal["provisional", "benchmarked"]
    accepted: Literal[True]
    hardware_profile_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    roles: tuple[StudentModelRole, ...] = Field(min_length=1)

    @field_validator(
        "recommendation_id",
        "hardware_profile_id",
        "model_id",
        "revision",
        "quantization",
        "runtime",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Recommendation reference fields must not be blank")
        return value

    @field_validator("path")
    @classmethod
    def require_json_path(cls, value: Path) -> Path:
        if value.suffix.lower() != ".json":
            raise ValueError("Recommendation path must use the .json extension")
        return value

    @field_validator("roles")
    @classmethod
    def require_unique_roles(cls, value: tuple[StudentModelRole, ...])-> tuple[StudentModelRole, ...]:
        if len(set(value)) != len(value):
            raise ValueError(
                "Student model roles must be unique"
            )
        return value


class StudentTarget(BaseModel):
    """One accepted Student configuration shared by a collection plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hardware: HardwareProfileReference
    recommendation: BaseModelRecommendationReference

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.recommendation.hardware_profile_id != self.hardware.profile_id:
            raise ValueError("Recommendation must reference the selected hardware profile")

        if EnvironmentPurpose.INFERENCE not in self.hardware.purposes:
            raise ValueError(
                "Student target hardaware must include the inference purpose"
            )
        return self


def _resolve_artifact(path: Path, artifact_root: Path | None) -> Path:
    if path.is_absolute():
        return path
    return (artifact_root or Path.cwd()) / path


def _load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Student artifact cannot be read: {path}") from error

    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Student artifact SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Student artifact is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Student artifact must contain a JSON object: {path}")
    return value


def _require_equal(payload: dict[str, Any], key: str, expected: object, label: str) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"{label} field {key!r} does not match the collection plan")

def _require_string_set(
        actual: object,
        expected: set[str],
        label: str,
) -> None:
    if not isinstance(actual, list):
        raise ValueError(
            f"{label} must be a JSON array"
        )

    if not all(isinstance(item, str) for item in actual or len(set(actual)) != len(actual)):
        raise ValueError(f"{label} must contain unique strings")

    if set(actual) != expected:
        raise ValueError(
            f"{label} does not match the collection plan"
        )

def _verify_hardware_profile(
        target: StudentTarget,
        *,
        artifact_root: Path | None,
) -> None:
    reference = target.hardware
    hardware_path = _resolve_artifact(reference.path, artifact_root,)
    hardware = _load_verified_json(hardware_path, reference.sha256,)

    _require_equal(hardware, "schema_version", "0.2", "Hardware profile")
    _require_equal(hardware, "profile_id", reference.profile_id, "Hardware profile")

    environment = hardware.get("environment")

    if not isinstance(environment, dict):
        raise ValueError("Hardware profile must contain an environment object")

    _require_equal(environment, "verification_status", reference.verification_status, "Hardware environment")
    _require_string_set(environment.get("purposes"), {purpose.value for purpose in reference.purposes}, "Hardware environment purpose")

def _verify_recommendation(target: StudentTarget, *, artifact_root: Path | None)->None:
    reference = target.recommendation
    recommendation_path = _resolve_artifact(reference.path, artifact_root)
    recommendation = _load_verified_json(recommendation_path, reference.sha256)
    _require_equal(recommendation, "schema_version", "0.1", "Recommendation")
    _require_equal(recommendation, "recommendation_id", reference.recommendation_id, "Recommendation")
    _require_equal(recommendation, "status", reference.evidence_status, "Recommendation")
    _require_equal(recommendation, "hardware_profile_id", target.hardware.profile_id, "Recommendation")

    primary = recommendation.get("primary")

    if not isinstance(primary, dict):
        raise ValueError("Recommendation must contain a primary model object")

    for key in ("model_id","revision","quantization", "runtime"):
        _require_equal(
            primary, key, getattr(reference, key), "Recommendation primary model"
        )

    _require_string_set(primary.get("roles"), {role.value for role in reference.roles}, "Recommendation primary model roles")

def verify_student_target(
    target: StudentTarget,
    *,
    artifact_root: Path | None = None,
) -> None:
    """Verify Student artifacts and their plan-critical fields."""

    _verify_hardware_profile(target, artifact_root = artifact_root,)
    _verify_recommendation(target, artifact_root=artifact_root)