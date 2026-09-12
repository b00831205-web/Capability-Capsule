"""Functional tests for dataset-scale and training-run provenance."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from capability_capsule.eval.curation import DuplicateTrajectory
from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
)
from capability_capsule.eval.tasks import (
    TaskCategory,
    TaskDifficulty,
    TaskSpec,
)
from capability_capsule.eval.training_provenance import (
    TrainingRunManifest,
    build_training_dataset_stats,
)


def make_task(
    task_id: str,
    split: DatasetSplit,
    category: TaskCategory,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        fixture_id=f"fixture-{task_id}",
        fixture_family_id=f"family-{task_id}",
        fixture_revision=f"revision-{task_id}",
        defect_family="smoke-defect",
        split=split,
        category=category,
        difficulty=TaskDifficulty.EASY,
        task=f"Complete {task_id}.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=(f"validator-{task_id}",),
    )


def make_trajectory(
    task: TaskSpec,
    trajectory_id: str,
    *,
    with_tool_call: bool = False,
) -> TeacherTrajectory:
    messages = (
        TrajectoryMessage(role=MessageRole.USER, content=task.task),
        TrajectoryMessage(
            role=MessageRole.ASSISTANT,
            content="Completed.",
            tool_calls=(
                (ToolCallRecord(name="exec_command"),)
                if with_tool_call
                else ()
            ),
        ),
    )
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=task.task_id,
        split=task.split,
        teacher_model="gpt-5",
        teacher_skill_version="0.1.0",
        task=task.task,
        messages=messages,
        source_revision=task.fixture_revision,
    )


def make_dataset_and_tasks() -> tuple[CuratedTeacherDataset, tuple[TaskSpec, ...]]:
    train_task = make_task(
        "task-train",
        DatasetSplit.TRAIN,
        TaskCategory.CODE_SEARCH,
    )
    validation_task = make_task(
        "task-validation",
        DatasetSplit.VALIDATION,
        TaskCategory.SINGLE_FILE_CHANGE,
    )
    dataset = CuratedTeacherDataset(
        train=(make_trajectory(train_task, "trajectory-train", with_tool_call=True),),
        validation=(make_trajectory(validation_task, "trajectory-validation"),),
        duplicates=(
            DuplicateTrajectory(
                fingerprint="a" * 64,
                kept_trajectory_id="trajectory-train",
                duplicate_trajectory_id="trajectory-discarded",
            ),
        ),
    )
    return dataset, (train_task, validation_task)


def test_build_training_dataset_stats_measures_curated_dataset() -> None:
    dataset, tasks = make_dataset_and_tasks()

    stats = build_training_dataset_stats(
        dataset,
        tasks=tasks,
        dataset_id="teacher-smoke-v1",
        dataset_digest="b" * 64,
    )

    assert stats.dataset_id == "teacher-smoke-v1"
    assert stats.dataset_digest == "b" * 64
    assert stats.discarded_duplicate_count == 1
    assert stats.train.trajectory_count == 1
    assert stats.train.message_count == 2
    assert stats.train.tool_call_count == 1
    assert stats.train.exact_token_count is None
    assert stats.train.tokenizer_id is None
    assert stats.validation.trajectory_count == 1
    assert stats.category_counts == {
        TaskCategory.CODE_SEARCH: 1,
        TaskCategory.SINGLE_FILE_CHANGE: 1,
    }


def test_build_training_dataset_stats_records_exact_tokens_with_tokenizer() -> None:
    dataset, tasks = make_dataset_and_tasks()

    stats = build_training_dataset_stats(
        dataset,
        tasks=tasks,
        dataset_id="teacher-smoke-v1",
        dataset_digest="b" * 64,
        tokenizer_id="test-tokenizer-v1",
        token_counter=lambda trajectory: len(trajectory.messages) * 10,
    )

    assert stats.train.exact_token_count == 20
    assert stats.validation.exact_token_count == 20
    assert stats.train.tokenizer_id == "test-tokenizer-v1"


@pytest.mark.parametrize(
    ("tokenizer_id", "token_counter"),
    [
        ("test-tokenizer-v1", None),
        (None, lambda trajectory: 1),
    ],
)
def test_build_training_dataset_stats_requires_tokenizer_pair(
    tokenizer_id: str | None,
    token_counter: Callable[[TeacherTrajectory], int] | None,
) -> None:
    dataset, tasks = make_dataset_and_tasks()

    with pytest.raises(ValueError, match="tokenizer_id and token_counter"):
        build_training_dataset_stats(
            dataset,
            tasks=tasks,
            dataset_id="teacher-smoke-v1",
            dataset_digest="b" * 64,
            tokenizer_id=tokenizer_id,
            token_counter=token_counter,
        )


def test_training_run_manifest_binds_training_to_dataset_stats() -> None:
    dataset, tasks = make_dataset_and_tasks()
    stats = build_training_dataset_stats(
        dataset,
        tasks=tasks,
        dataset_id="teacher-smoke-v1",
        dataset_digest="b" * 64,
    )

    manifest = TrainingRunManifest(
        run_id="training-run-001",
        output_capsule_id="capsule-smoke-v1",
        dataset_stats=stats,
        base_model_id="base-model-v1",
        trainer_id="lora-trainer-v1",
        hardware_id="gpu-lab-01",
        random_seed=42,
        started_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        hyperparameters={"learning_rate": 0.0001, "epochs": 3},
    )

    assert manifest.dataset_id == "teacher-smoke-v1"
    assert manifest.dataset_digest == "b" * 64
    assert manifest.model_dump_json()


def test_training_run_manifest_requires_timezone_aware_start_time() -> None:
    dataset, tasks = make_dataset_and_tasks()
    stats = build_training_dataset_stats(
        dataset,
        tasks=tasks,
        dataset_id="teacher-smoke-v1",
        dataset_digest="b" * 64,
    )

    with pytest.raises(ValidationError, match="timezone"):
        TrainingRunManifest(
            run_id="training-run-001",
            output_capsule_id="capsule-smoke-v1",
            dataset_stats=stats,
            base_model_id="base-model-v1",
            trainer_id="lora-trainer-v1",
            hardware_id="gpu-lab-01",
            random_seed=42,
            started_at=datetime(2026, 9, 12),
        )
