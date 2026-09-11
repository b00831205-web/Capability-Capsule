"""Functional tests for versioned training and experiment records."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from capability_capsule.eval.records import (
    CaseResult,
    DatasetSplit,
    ExperimentManifest,
    MessageRole,
    RunStatus,
    StudyKind,
    TeacherTrajectory,
    TrajectoryMessage,
)


def test_teacher_trajectory_round_trips_as_json() -> None:
    record = TeacherTrajectory(
        trajectory_id="trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TRAIN,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task="Find the CLI entry point and explain it.",
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

    restored = TeacherTrajectory.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.schema_version == "0.1"


def test_primary_manifest_requires_at_least_thirty_repetitions() -> None:
    with pytest.raises(ValidationError):
        ExperimentManifest(
            experiment_id="experiment-001",
            study_kind=StudyKind.PRIMARY,
            created_at=datetime(2026, 9, 10, tzinfo=UTC),
            candidate_models=("model-0.8b", "model-1.5b"),
            repetitions=29,
            cycle_task_count=30,
            memory_budget_mb=8192,
            random_seed=42,
            held_out_dataset_id="held-out-v1",
        )


def test_pilot_manifest_can_use_fewer_repetitions() -> None:
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

    assert manifest.repetitions == 3


def test_completed_case_result_keeps_measurements() -> None:
    result = CaseResult(
        experiment_id="experiment-001",
        run_id="run-001",
        case_id="case-001",
        model_id="model-0.8b+lora",
        condition_id="lora-no-rag",
        hardware_id="laptop-v1",
        repetition=1,
        cycle_position=4,
        status=RunStatus.COMPLETED,
        success=True,
        time_bounded_success=True,
        score=0.9,
        started_at=datetime(2026, 9, 10, tzinfo=UTC),
        duration_ms=1250.0,
        input_tokens=200,
        output_tokens=80,
        peak_rss_mb=2048.0,
        tool_call_count=2,
        invalid_tool_call_count=0,
    )

    assert result.score == pytest.approx(0.9)
    assert result.status is RunStatus.COMPLETED


@pytest.mark.parametrize("blank_value", ["", "   "])
def test_record_identifiers_must_not_be_blank(blank_value: str) -> None:
    with pytest.raises(ValidationError):
        TeacherTrajectory(
            trajectory_id=blank_value,
            task_id="task-001",
            split=DatasetSplit.TRAIN,
            teacher_model="teacher",
            teacher_skill_version="0.1.0",
            task="Do the task.",
            messages=(
                TrajectoryMessage(role=MessageRole.USER, content="Start."),
                TrajectoryMessage(
                    role=MessageRole.ASSISTANT,
                    content="Done.",
                ),
            ),
        )
