"""Functional tests for validating Teacher trajectories against task specs."""

from datetime import timedelta

import pytest

from capability_capsule.eval.dataset_validation import validate_teacher_dataset
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


def make_task(
    task_id: str,
    *,
    fixture_id: str = "fixture-001",
    fixture_family_id: str = "fixture-family-001",
    defect_family: str = "cli-entry-point",
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        fixture_id=fixture_id,
        fixture_family_id=fixture_family_id,
        fixture_revision="fixture-sha-001",
        defect_family=defect_family,
        split=split,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Find the CLI entry point and cite its path.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("exact-path:cli-entry-point",),
    )


def make_trajectory(
    trajectory_id: str,
    *,
    task_id: str,
    split: DatasetSplit = DatasetSplit.TRAIN,
    source_revision: str = "fixture-sha-001",
    tool_name: str = "exec_command",
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=split,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task="Find the CLI entry point and cite its path.",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Find the CLI entry point.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="I will inspect the project.",
                tool_calls=(
                    ToolCallRecord(
                        name=tool_name,
                        arguments={"cmd": "rg entry"},
                    ),
                ),
            ),
            TrajectoryMessage(
                role=MessageRole.TOOL,
                content="src/capability_capsule/cli/__init__.py",
                tool_name=tool_name,
            ),
        ),
        source_revision=source_revision,
    )


def test_valid_teacher_dataset_matches_task_contracts() -> None:
    task = make_task("task-001")
    trajectory = make_trajectory("trajectory-001", task_id="task-001")

    validate_teacher_dataset((trajectory,), tasks=(task,))


def test_teacher_dataset_rejects_unknown_task() -> None:
    trajectory = make_trajectory("trajectory-001", task_id="missing-task")

    with pytest.raises(ValueError, match="missing-task"):
        validate_teacher_dataset((trajectory,), tasks=())


def test_teacher_dataset_rejects_test_trajectory() -> None:
    task = make_task("task-001", split=DatasetSplit.TEST)
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TEST,
    )

    with pytest.raises(ValueError, match="test"):
        validate_teacher_dataset((trajectory,), tasks=(task,))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", DatasetSplit.VALIDATION),
        ("source_revision", "wrong-revision"),
        ("task", "A different task."),
    ],
)
def test_teacher_dataset_rejects_contract_mismatch(
    field: str,
    value: object,
) -> None:
    task = make_task("task-001")
    payload = make_trajectory(
        "trajectory-001",
        task_id="task-001",
    ).model_dump()
    payload[field] = value
    trajectory = TeacherTrajectory.model_validate(payload)

    with pytest.raises(ValueError, match="task-001"):
        validate_teacher_dataset((trajectory,), tasks=(task,))


def test_teacher_dataset_rejects_disallowed_tool() -> None:
    task = make_task("task-001")
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_name="web_search",
    )

    with pytest.raises(ValueError, match="web_search"):
        validate_teacher_dataset((trajectory,), tasks=(task,))


def test_teacher_dataset_rejects_duplicate_trajectory_ids() -> None:
    task = make_task("task-001")
    first = make_trajectory("trajectory-001", task_id="task-001")
    second = make_trajectory("trajectory-001", task_id="task-001")

    with pytest.raises(ValueError, match="trajectory-001"):
        validate_teacher_dataset((first, second), tasks=(task,))


def test_teacher_dataset_rejects_group_leakage_across_splits() -> None:
    train_task = make_task("task-train")
    validation_task = make_task(
        "task-validation",
        fixture_id="fixture-renamed-copy",
        split=DatasetSplit.VALIDATION,
    )
    train = make_trajectory("trajectory-train", task_id="task-train")
    validation = make_trajectory(
        "trajectory-validation",
        task_id="task-validation",
        split=DatasetSplit.VALIDATION,
    )

    with pytest.raises(ValueError, match="fixture-family-001:cli-entry-point"):
        validate_teacher_dataset(
            (train, validation),
            tasks=(train_task, validation_task),
        )
