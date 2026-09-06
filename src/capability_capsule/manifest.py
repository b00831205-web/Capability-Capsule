"""Versioned capsule manifest and artifact provenance contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(StrEnum):
    """Supported origins for capsule artifacts."""

    REPO = "repo"
    LOCAL_DOCUMENT = "local_document"
    PREFETCHED = "prefetched"


class CreatedBy(StrEnum):
    """Agent responsible for creating an artifact."""

    ORIGINAL = "original"
    CLOUD_TEACHER = "cloud_teacher"


class ArtifactProvenance(BaseModel):
    """Traceable origin information for one capsule artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: SourceType
    source: str = Field(min_length=1)
    created_by: CreatedBy
    capsule_build_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str | None = None


class CapsuleManifest(BaseModel):
    """Versioned description of a generated capability capsule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    capsule_build_id: UUID = Field(default_factory=uuid4)
    task: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    size_budget_bytes: int = Field(gt=0)
    offline_duration_hours: float = Field(gt=0)
    generation_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    artifacts: tuple[ArtifactProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_build_ids(self) -> Self:
        """Ensure every artifact belongs to this capsule build."""

        mismatched = [
            artifact
            for artifact in self.artifacts
            if artifact.capsule_build_id != self.capsule_build_id
        ]
        if mismatched:
            raise ValueError("artifact capsule_build_id does not match manifest")
        return self
