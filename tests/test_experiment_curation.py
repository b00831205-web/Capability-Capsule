"""Functional tests for Teacher trajectory dataset curation."""

from capability_capsule.eval.curation import (
    deduplicate_trajectories,
    partition_trajectories,
    trajectory_fingerprint,
    validate_split_isolation,
)
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
)


def make_trajectory(
    trajectory_id: str,
    *,
    task_id: str,
    split: DatasetSplit,
    task: str = "Inspect the project and explain the CLI entry point.",
    command: str = "rg entry",
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=split,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        task=task,
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Inspect the project.\r\n",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="I will locate the entry point.",
                tool_calls=(
                    ToolCallRecord(
                        name="exec_command",
                        arguments={"cmd": command},
                    ),
                ),
            ),
            TrajectoryMessage(
                role=MessageRole.TOOL,
                content="src/capability_capsule/cli/__init__.py   \r\n",
                tool_name="exec_command",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="The CLI entry point is capability_capsule.cli:app.",
            ),
        ),
        source_revision="255fe0b",
    )


def test_fingerprint_normalizes_task_and_line_endings() -> None:
    first = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TRAIN,
    )
    equivalent = make_trajectory(
        "trajectory-002",
        task_id="task-002",
        split=DatasetSplit.VALIDATION,
        task="  Inspect   the project and explain the CLI entry point.  ",
    )

    assert trajectory_fingerprint(first) == trajectory_fingerprint(equivalent)


def test_fingerprint_keeps_meaningful_tool_arguments() -> None:
    first = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TRAIN,
        command="rg entry",
    )
    changed = make_trajectory(
        "trajectory-002",
        task_id="task-002",
        split=DatasetSplit.TRAIN,
        command="rg config",
    )

    assert trajectory_fingerprint(first) != trajectory_fingerprint(changed)


def test_deduplication_preserves_first_record_and_reports_duplicate() -> None:
    first = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TRAIN,
    )
    duplicate = make_trajectory(
        "trajectory-002",
        task_id="task-002",
        split=DatasetSplit.TRAIN,
    )

    result = deduplicate_trajectories((first, duplicate))

    assert result.unique_trajectories == (first,)
    assert len(result.duplicates) == 1
    assert result.duplicates[0].kept_trajectory_id == "trajectory-001"
    assert result.duplicates[0].duplicate_trajectory_id == "trajectory-002"
    assert result.duplicates[0].fingerprint == trajectory_fingerprint(first)


def test_partition_trajectories_preserves_order_within_each_split() -> None:
    validation = make_trajectory(
        "trajectory-validation",
        task_id="task-validation",
        split=DatasetSplit.VALIDATION,
    )
    train_first = make_trajectory(
        "trajectory-train-001",
        task_id="task-train-001",
        split=DatasetSplit.TRAIN,
    )
    train_second = make_trajectory(
        "trajectory-train-002",
        task_id="task-train-002",
        split=DatasetSplit.TRAIN,
        command="rg config",
    )

    partitions = partition_trajectories((validation, train_first, train_second))

    assert partitions[DatasetSplit.TRAIN] == (train_first, train_second)
    assert partitions[DatasetSplit.VALIDATION] == (validation,)
    assert partitions[DatasetSplit.TEST] == ()


def test_split_isolation_rejects_one_group_across_splits() -> None:
    train = make_trajectory(
        "trajectory-train",
        task_id="task-train",
        split=DatasetSplit.TRAIN,
    )
    test = make_trajectory(
        "trajectory-test",
        task_id="task-test",
        split=DatasetSplit.TEST,
        command="rg config",
    )

    group_by_task = {
        "task-train": "fixture-001:defect-family-a",
        "task-test": "fixture-001:defect-family-a",
    }

    try:
        validate_split_isolation((train, test), group_by_task=group_by_task)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected split leakage to be rejected")

    assert "fixture-001:defect-family-a" in message
    assert "train" in message
    assert "test" in message


def test_split_isolation_requires_a_group_for_every_task() -> None:
    record = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        split=DatasetSplit.TRAIN,
    )

    try:
        validate_split_isolation((record,), group_by_task={})
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected missing grouping metadata to be rejected")

    assert "task-001" in message
