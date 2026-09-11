"""Immutable publication of curated Teacher datasets."""

import shutil
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset


_ARTIFACT_FILENAMES = (
    "train.jsonl",
    "validation.jsonl",
    "duplicates.jsonl"
)


def _validate_dataset_id(dataset_id: str) -> str:
    if (
        not dataset_id.strip()
        or dataset_id != dataset_id.strip()
        or dataset_id in {".", ".."}
        or "/" in dataset_id
        or "\\" in dataset_id
        or "\0" in dataset_id
    ):
        raise ValueError(
            "dataset_id must be a safe, non-blank directory name"
        )

    return dataset_id


class PublishedArtifact(BaseModel):
    """Integrity metadata for one published dataset file."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    filename: Literal[
        "train.jsonl",
        "validation.jsonl",
        "duplicates.jsonl"
    ]
    record_count: int = Field(ge = 0)
    byte_count: int = Field(ge = 0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class DatasetPublicationManifest(BaseModel):
    """Portable manifest describing one immutable dataset version."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    dataset_id: str = Field(min_length = 1)
    artifacts: tuple[PublishedArtifact, ...] = Field(
        min_length=3,
        max_length = 3,
    )

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        return _validate_dataset_id(value)

    @field_validator("artifacts")
    @classmethod
    def require_expected_artifacts(
        cls,
        value: tuple[PublishedArtifact, ...],
    ) -> tuple[PublishedArtifact, ...]:
        filenames = tuple(artifact.filename for artifact in value)

        if filenames != _ARTIFACT_FILENAMES:
            raise ValueError(
                "artifacts must contain train, validation, and duplicates "
                "in canonical order"
            )

        return value


def _write_jsonl(
        path: Path,
        records: Sequence[BaseModel],
) -> PublishedArtifact:
    digest = sha256()
    byte_count = 0

    with path.open("xb") as stream:
        for record in records:
            encoded = record.model_dump_json().encode("utf-8") + b"\n"
            stream.write(encoded)
            digest.update(encoded)
            byte_count += len(encoded)

    return PublishedArtifact(
        filename = cast(Literal["train.jsonl", "validation.jsonl", "duplicates.jsonl"], path.name),
        record_count = len(records),
        byte_count = byte_count,
        sha256 = digest.hexdigest(),
    )

def publish_teacher_dataset(
        dataset: CuratedTeacherDataset,
        *,
        output_root: Path,
        dataset_id: str,
) -> DatasetPublicationManifest:
    """Publish a curated dataset withou overwriting an existing version."""

    validated_dataset_id = _validate_dataset_id(dataset_id)
    output_root = Path(output_root)
    version_dir = output_root / validated_dataset_id

    if version_dir.exists():
        raise FileExistsError(
            f"Dataset version {validated_dataset_id!r} already exists"
        )

    output_root.mkdir(parents=True, exist_ok = True)

    staging_dir = Path(
        mkdtemp(
            prefix = f".{validated_dataset_id}",
            dir = output_root,
        )
    )

    try:
        artifacts = (
            _write_jsonl(
                staging_dir / "train.jsonl",
                dataset.train,
            ),
            _write_jsonl(
                staging_dir / "validation.jsonl",
                dataset.validation,
            ),
            _write_jsonl(
                staging_dir / "duplicates.jsonl",
                dataset.duplicates,
            ),
        )

        manifest = DatasetPublicationManifest(
            dataset_id = validated_dataset_id,
            artifacts = artifacts,
        )

        manifest_bytes = (
            manifest.model_dump_json(indent = 2).encode("utf-8")+b"\n"
        )
        (staging_dir / "manifest.json").write_bytes(manifest_bytes)

        staging_dir.rename(version_dir)
        return manifest

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)