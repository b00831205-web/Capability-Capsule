"""Functional tests for experiment JSONL persistence."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    ExperimentManifest,
    MessageRole,
    StudyKind,
    TeacherTrajectory,
    TrajectoryMessage,
)


def make_trajectory(trajectory_id: str) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=f"task-for-{trajectory_id}",
        split=DatasetSplit.TRAIN,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task="Inspect the project and explain its CLI entry point.",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Inspect the project.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="The CLI entry point is capability_capsule.cli:app.",
            ),
        ),
        source_revision="255fe0b",
    )


def test_append_jsonl_creates_parent_and_round_trips_records(tmp_path: Path) -> None:
    output_path = tmp_path / "datasets" / "teacher-trajectories.jsonl"
    first = make_trajectory("trajectory-001")
    second = make_trajectory("trajectory-002")

    append_jsonl(output_path, first)
    append_jsonl(output_path, second)

    restored = load_jsonl(output_path, TeacherTrajectory)

    assert restored == (first, second)
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_load_jsonl_returns_empty_tuple_for_empty_file(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.touch()

    assert load_jsonl(input_path, TeacherTrajectory) == ()


def test_load_jsonl_validates_records_as_requested_model(tmp_path: Path) -> None:
    input_path = tmp_path / "manifests.jsonl"
    manifest = ExperimentManifest(
        experiment_id="pilot-001",
        study_kind=StudyKind.PILOT,
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
        candidate_models=("model-0.8b",),
        repetitions=3,
        cycle_task_count=10,
        memory_budget_mb=8192,
        random_seed=42,
        held_out_dataset_id="held-out-v1",
    )
    append_jsonl(input_path, manifest)

    assert load_jsonl(input_path, ExperimentManifest) == (manifest,)


def test_load_jsonl_reports_the_path_and_bad_line_number(tmp_path: Path) -> None:
    input_path = tmp_path / "teacher-trajectories.jsonl"
    first = make_trajectory("trajectory-001")
    input_path.write_text(
        f"{first.model_dump_json()}\n{{not valid json}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_jsonl(input_path, TeacherTrajectory)

    message = str(error.value)
    assert str(input_path) in message
    assert "line 2" in message


def test_load_jsonl_rejects_blank_lines_with_location(tmp_path: Path) -> None:
    input_path = tmp_path / "teacher-trajectories.jsonl"
    first = make_trajectory("trajectory-001")
    input_path.write_text(
        f"{first.model_dump_json()}\n\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(input_path, TeacherTrajectory)
