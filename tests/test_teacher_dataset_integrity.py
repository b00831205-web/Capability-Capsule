"""Functional tests for loading and verifying published Teacher datasets."""

from pathlib import Path

import pytest

from capability_capsule.eval.dataset_integrity import (
    load_teacher_dataset_publication,
)
from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    TrajectoryMessage,
)
from capability_capsule.eval.teacher_collection_plan_publication import (
    load_teacher_collection_plan,
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


def publish_fixture(tmp_path: Path) -> tuple[Path, CuratedTeacherDataset]:
    dataset = CuratedTeacherDataset(
        train=(make_trajectory("train-001", DatasetSplit.TRAIN),),
        validation=(
            make_trajectory("validation-001", DatasetSplit.VALIDATION),
        ),
        duplicates=(),
    )
    publish_teacher_dataset(
        dataset,
        output_root=tmp_path,
        dataset_id="teacher-pilot-v1",
    )
    return tmp_path / "teacher-pilot-v1", dataset


def test_load_teacher_dataset_publication_verifies_and_loads_records(
    tmp_path: Path,
) -> None:
    version_dir, expected_dataset = publish_fixture(tmp_path)

    loaded = load_teacher_dataset_publication(version_dir)

    assert loaded.manifest.dataset_id == "teacher-pilot-v1"
    assert loaded.dataset == expected_dataset


def test_load_teacher_dataset_publication_rejects_modified_artifact(
    tmp_path: Path,
) -> None:
    version_dir, _ = publish_fixture(tmp_path)
    train_path = version_dir / "train.jsonl"
    train_path.write_bytes(train_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=r"train\.jsonl.*(byte count|SHA-256)"):
        load_teacher_dataset_publication(version_dir)


def test_load_teacher_dataset_publication_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    version_dir, _ = publish_fixture(tmp_path)
    (version_dir / "validation.jsonl").unlink()

    with pytest.raises(ValueError, match=r"missing.*validation\.jsonl"):
        load_teacher_dataset_publication(version_dir)


def test_load_teacher_dataset_publication_rejects_unexpected_file(
    tmp_path: Path,
) -> None:
    version_dir, _ = publish_fixture(tmp_path)
    (version_dir / "notes.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected.*notes\.txt"):
        load_teacher_dataset_publication(version_dir)


def test_load_teacher_dataset_publication_rejects_wrong_record_count(
    tmp_path: Path,
) -> None:
    version_dir, _ = publish_fixture(tmp_path)
    manifest_path = version_dir / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace('"record_count": 1', '"record_count": 2', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"train\.jsonl.*record count"):
        load_teacher_dataset_publication(version_dir)


def test_load_teacher_dataset_publication_rejects_wrong_partition(
    tmp_path: Path,
) -> None:
    dataset = CuratedTeacherDataset(
        train=(make_trajectory("wrong-split", DatasetSplit.VALIDATION),),
        validation=(),
        duplicates=(),
    )
    publish_teacher_dataset(
        dataset,
        output_root=tmp_path,
        dataset_id="teacher-wrong-split-v1",
    )

    with pytest.raises(ValueError, match=r"train\.jsonl.*validation"):
        load_teacher_dataset_publication(
            tmp_path / "teacher-wrong-split-v1"
        )


def test_load_teacher_dataset_publication_rejects_wrong_directory_name(
    tmp_path: Path,
) -> None:
    version_dir, _ = publish_fixture(tmp_path)
    renamed_dir = tmp_path / "renamed-dataset"
    version_dir.rename(renamed_dir)

    with pytest.raises(ValueError, match="dataset_id"):
        load_teacher_dataset_publication(renamed_dir)


def test_checked_in_stage1_publication_is_complete_and_harness_aligned() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    verified = load_teacher_dataset_publication(
        repository_root
        / "datasets"
        / "teacher"
        / "published"
        / "stage1-codex-001"
    )
    plan = load_teacher_collection_plan(
        repository_root / "plans" / "teacher" / "stage1-codex-001"
    ).plan
    profile = verify_harness_profile(
        plan.harness_profile,
        artifact_root=repository_root,
    )
    trajectories = verified.dataset.train + verified.dataset.validation

    validate_teacher_dataset(
        trajectories,
        tasks=tuple(assignment.task for assignment in plan.assignments),
        harness_profile=profile,
    )

    assert verified.manifest.dataset_id == "stage1-codex-001"
    assert len(verified.dataset.train) == 8
    assert len(verified.dataset.validation) == 2
    assert verified.dataset.duplicates == ()
    assert len({record.trajectory_id for record in trajectories}) == 10


def test_checked_in_stage1_powershell_publication_combines_increment() -> None:
    root = Path(__file__).resolve().parents[1]
    verified = load_teacher_dataset_publication(
        root
        / "datasets"
        / "teacher"
        / "published"
        / "stage1-codex-powershell-002"
    )
    original_plan = load_teacher_collection_plan(
        root / "plans" / "teacher" / "stage1-codex-001"
    ).plan
    increment_plan = load_teacher_collection_plan(
        root / "plans" / "teacher" / "stage1-codex-powershell-002"
    ).plan
    profile = verify_harness_profile(
        increment_plan.harness_profile,
        artifact_root=root,
    )
    trajectories = verified.dataset.train + verified.dataset.validation

    validate_teacher_dataset(
        trajectories,
        tasks=tuple(
            assignment.task
            for assignment in original_plan.assignments
            + increment_plan.assignments
        ),
        harness_profile=profile,
    )

    assert len(verified.dataset.train) == 16
    assert len(verified.dataset.validation) == 2
    assert verified.dataset.duplicates == ()
    assert len({record.trajectory_id for record in trajectories}) == 18


def test_checked_in_powershell_edit_publication_combines_increment() -> None:
    root = Path(__file__).resolve().parents[1]
    verified = load_teacher_dataset_publication(
        root
        / "datasets"
        / "teacher"
        / "published"
        / "stage1-codex-powershell-edit-003"
    )
    base_plan = load_teacher_collection_plan(
        root / "plans" / "teacher" / "stage1-codex-001"
    ).plan
    powershell_plan = load_teacher_collection_plan(
        root / "plans" / "teacher" / "stage1-codex-powershell-002"
    ).plan
    edit_plan = load_teacher_collection_plan(
        root / "plans" / "teacher" / "stage1-codex-powershell-edit-003"
    ).plan
    profile = verify_harness_profile(
        edit_plan.harness_profile,
        artifact_root=root,
    )
    trajectories = verified.dataset.train + verified.dataset.validation

    validate_teacher_dataset(
        trajectories,
        tasks=tuple(
            assignment.task
            for assignment in (
                base_plan.assignments
                + powershell_plan.assignments
                + edit_plan.assignments
            )
        ),
        harness_profile=profile,
    )

    assert verified.manifest.dataset_id == "stage1-codex-powershell-edit-003"
    assert len(verified.dataset.train) == 24
    assert len(verified.dataset.validation) == 2
    assert verified.dataset.duplicates == ()
    assert len({record.trajectory_id for record in trajectories}) == 26
