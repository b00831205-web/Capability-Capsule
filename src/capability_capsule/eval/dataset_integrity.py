"""Integrity verification for published Teacher datasets."""

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from capability_capsule.eval.curation import DuplicateTrajectory
from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.dataset_publication import (
    DatasetPublicationManifest,
    PublishedArtifact,
)
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory


_EXPECTED_FILENAMES = {
    "manifest.json",
    "train.jsonl",
    "validation.jsonl",
    "duplicates.jsonl"
}


class VerifiedTeacherDataset(BaseModel):
    """A published dataset loaded after all integrity checks pass."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    manifest: DatasetPublicationManifest
    dataset: CuratedTeacherDataset


def _validate_file_set(version_dir: Path) -> None:
    actual = {entry.name for entry in version_dir.iterdir()}
    missing = _EXPECTED_FILENAMES - actual
    unexpected = actual - _EXPECTED_FILENAMES

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Published dataset has missing files: {names}")

    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Published dataset has unexpected files: {names}")

def _verify_artifact(
        version_dir: Path,
        artifact: PublishedArtifact,
) -> None:
    artifact_path = version_dir / artifact.filename
    payload = artifact_path.read_bytes()
    actual_byte_count = len(payload)

    if actual_byte_count != artifact.byte_count:
        raise ValueError(
            f"{artifact.filename} byte count mismatch: "
            f"expected {artifact.byte_count}, got {actual_byte_count}"
        )

    actual_digest = sha256(payload).hexdigest()

    if actual_digest != artifact.sha256:
        raise ValueError(
            f"{artifact.filename} SHA-256 mismatch: "
            f"expected {artifact.sha256}, got {actual_digest}"
        )

def _validate_record_count(
        artifact: PublishedArtifact,
        records: tuple[BaseModel, ...],
) -> None:
    actual_count = len(records)

    if actual_count != artifact.record_count:
        raise ValueError(
            f"{artifact.filename} record count mismatch: "
            f"expected {artifact.record_count}, got {actual_count}"
        )

def _validate_partition(
        filename: str,
        trajectories: tuple[TeacherTrajectory, ...],
        expected_split: DatasetSplit,
) -> None:
    for trajectory in trajectories:
        if trajectory.split is not expected_split:
            raise ValueError(
                f"{filename} contains {trajectory.split.value} trajectory "
                f"{trajectory.trajectory_id!r}"
            )

def load_teacher_dataset_publication(
        version_dir: Path
) -> VerifiedTeacherDataset:
    """Load a published dataset after verifying its manifest and artifacts."""

    version_dir = Path(version_dir)

    if not version_dir.is_dir():
        raise ValueError(
            f"Published dataset directory does not exist: {version_dir}"
        )

    _validate_file_set(version_dir)

    manifest_path = version_dir / "manifest.json"

    try:
        manifest = DatasetPublicationManifest.model_validate_json(
            manifest_path.read_text(encoding = "utf-8")
        )

    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid publication manifest: {manifest_path}"
        ) from error

    if manifest.dataset_id != version_dir.name:
        raise ValueError(
            f"Manifest dataset_id {manifest.dataset_id!r} does not match "
            f"directory name {version_dir.name!r}"
        )

    artifacts = {
        artifact.filename: artifact
        for artifact in manifest.artifacts
    }

    for artifact in manifest.artifacts:
        _verify_artifact(version_dir, artifact)

    train = load_jsonl(
        version_dir / "train.jsonl",
        TeacherTrajectory,
    )
    validation = load_jsonl(
        version_dir / "validation.jsonl",
        TeacherTrajectory,
    )
    duplicates = load_jsonl(
        version_dir / "duplicates.jsonl",
        DuplicateTrajectory,
    )
    _validate_record_count(artifacts["train.jsonl"], train)
    _validate_record_count(
        artifacts["validation.jsonl"],
        validation,
    )
    _validate_record_count(
        artifacts["duplicates.jsonl"],
        duplicates,
    )

    _validate_partition(
        "train.jsonl",
        train,
        DatasetSplit.TRAIN,
    )
    _validate_partition(
        "validation.jsonl",
        validation,
        DatasetSplit.VALIDATION,
    )

    dataset = CuratedTeacherDataset(
        train = train,
        validation = validation,
        duplicates = duplicates,
    )

    return VerifiedTeacherDataset(
        manifest = manifest,
        dataset = dataset
    )