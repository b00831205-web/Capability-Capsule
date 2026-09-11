"""Functional tests for immutable Teacher dataset publication."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from capability_capsule.eval.curation import DuplicateTrajectory
from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.dataset_publication import (
    DatasetPublicationManifest,
    publish_teacher_dataset,
)
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    TrajectoryMessage,
)


def make_trajectory(
    trajectory_id: str,
    split: DatasetSplit,
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=f"task-{trajectory_id}",
        split=split,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task=f"Complete {trajectory_id}.",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content=f"Complete {trajectory_id}.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content=f"Completed {trajectory_id}.",
            ),
        ),
        source_revision="fixture-sha-001",
    )


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_publish_teacher_dataset_writes_auditable_immutable_version(
    tmp_path: Path,
) -> None:
    train = make_trajectory("train-001", DatasetSplit.TRAIN)
    validation = make_trajectory("validation-001", DatasetSplit.VALIDATION)
    duplicate = DuplicateTrajectory(
        fingerprint="a" * 64,
        kept_trajectory_id="train-001",
        duplicate_trajectory_id="train-duplicate-001",
    )
    dataset = CuratedTeacherDataset(
        train=(train,),
        validation=(validation,),
        duplicates=(duplicate,),
    )

    manifest = publish_teacher_dataset(
        dataset,
        output_root=tmp_path,
        dataset_id="teacher-pilot-v1",
    )

    version_dir = tmp_path / "teacher-pilot-v1"
    manifest_path = version_dir / "manifest.json"
    stored_manifest = DatasetPublicationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest == stored_manifest
    assert manifest.dataset_id == "teacher-pilot-v1"
    assert load_jsonl(version_dir / "train.jsonl", TeacherTrajectory) == (train,)
    assert load_jsonl(
        version_dir / "validation.jsonl",
        TeacherTrajectory,
    ) == (validation,)
    assert load_jsonl(
        version_dir / "duplicates.jsonl",
        DuplicateTrajectory,
    ) == (duplicate,)

    for artifact in manifest.artifacts:
        artifact_path = version_dir / artifact.filename
        assert artifact.byte_count == artifact_path.stat().st_size
        assert artifact.sha256 == file_sha256(artifact_path)

    assert [(artifact.filename, artifact.record_count) for artifact in manifest.artifacts] == [
        ("train.jsonl", 1),
        ("validation.jsonl", 1),
        ("duplicates.jsonl", 1),
    ]


def test_publish_teacher_dataset_materializes_empty_partitions(tmp_path: Path) -> None:
    manifest = publish_teacher_dataset(
        CuratedTeacherDataset(train=(), validation=(), duplicates=()),
        output_root=tmp_path,
        dataset_id="teacher-empty-v1",
    )

    version_dir = tmp_path / "teacher-empty-v1"
    empty_digest = sha256(b"").hexdigest()

    for artifact in manifest.artifacts:
        artifact_path = version_dir / artifact.filename
        assert artifact_path.read_bytes() == b""
        assert artifact.record_count == 0
        assert artifact.byte_count == 0
        assert artifact.sha256 == empty_digest


def test_publish_teacher_dataset_refuses_to_overwrite_a_version(tmp_path: Path) -> None:
    version_dir = tmp_path / "teacher-pilot-v1"
    version_dir.mkdir()
    sentinel = version_dir / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="teacher-pilot-v1"):
        publish_teacher_dataset(
            CuratedTeacherDataset(train=(), validation=(), duplicates=()),
            output_root=tmp_path,
            dataset_id="teacher-pilot-v1",
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert set(version_dir.iterdir()) == {sentinel}


@pytest.mark.parametrize("dataset_id", ["", " ", ".", "..", "../escape", "nested/version"])
def test_publish_teacher_dataset_rejects_unsafe_dataset_ids(
    tmp_path: Path,
    dataset_id: str,
) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        publish_teacher_dataset(
            CuratedTeacherDataset(train=(), validation=(), duplicates=()),
            output_root=tmp_path,
            dataset_id=dataset_id,
        )

    assert list(tmp_path.iterdir()) == []


def test_publication_manifest_is_plain_json_without_absolute_paths(tmp_path: Path) -> None:
    publish_teacher_dataset(
        CuratedTeacherDataset(train=(), validation=(), duplicates=()),
        output_root=tmp_path,
        dataset_id="teacher-portable-v1",
    )

    manifest_path = tmp_path / "teacher-portable-v1" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["dataset_id"] == "teacher-portable-v1"
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")
