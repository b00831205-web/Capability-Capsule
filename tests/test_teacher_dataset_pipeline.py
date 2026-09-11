"""Functional tests for the Teacher dataset curation pipeline."""

from datetime import timedelta

from capability_capsule.eval.dataset_pipeline import curate_teacher_dataset
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    TrajectoryMessage,
)
from capability_capsule.eval.tasks import (
    TaskCategory,
    TaskDifficulty,
    TaskSpec,
)


def make_task(task_id: str, fixture_id: str, split: DatasetSplit) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        fixture_id=fixture_id,
        fixture_family_id=f"family-{fixture_id}",
        fixture_revision="fixture-sha-001",
        defect_family="code-location",
        split=split,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Find the CLI entry point.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("exact-path:cli-entry-point",),
    )


def make_trajectory(
    trajectory_id: str,
    task_id: str,
    split: DatasetSplit,
    *,
    answer: str = "The entry point is capability_capsule.cli:app.",
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=split,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task="Find the CLI entry point.",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Find the CLI entry point.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content=answer,
            ),
        ),
        source_revision="fixture-sha-001",
    )


def test_curate_teacher_dataset_validates_deduplicates_and_partitions() -> None:
    train_task = make_task("task-train", "fixture-train", DatasetSplit.TRAIN)
    validation_task = make_task(
        "task-validation",
        "fixture-validation",
        DatasetSplit.VALIDATION,
    )
    train = make_trajectory(
        "trajectory-train",
        "task-train",
        DatasetSplit.TRAIN,
    )
    duplicate = make_trajectory(
        "trajectory-validation",
        "task-validation",
        DatasetSplit.VALIDATION,
    )

    curated = curate_teacher_dataset(
        (train, duplicate),
        tasks=(train_task, validation_task),
    )

    assert curated.train == (train,)
    assert curated.validation == ()
    assert len(curated.duplicates) == 1
    assert curated.duplicates[0].kept_trajectory_id == "trajectory-train"
    assert curated.duplicates[0].duplicate_trajectory_id == "trajectory-validation"


def test_curate_teacher_dataset_preserves_distinct_records() -> None:
    train_task = make_task("task-train", "fixture-train", DatasetSplit.TRAIN)
    validation_task = make_task(
        "task-validation",
        "fixture-validation",
        DatasetSplit.VALIDATION,
    )
    train = make_trajectory(
        "trajectory-train",
        "task-train",
        DatasetSplit.TRAIN,
    )
    validation = make_trajectory(
        "trajectory-validation",
        "task-validation",
        DatasetSplit.VALIDATION,
        answer="The entry point is declared in pyproject.toml.",
    )

    curated = curate_teacher_dataset(
        (train, validation),
        tasks=(train_task, validation_task),
    )

    assert curated.train == (train,)
    assert curated.validation == (validation,)
    assert curated.duplicates == ()
    assert curated.schema_version == "0.1"
