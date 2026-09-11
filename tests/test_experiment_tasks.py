"""Functional tests for versioned task specifications and logical cycles."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.tasks import (
    ChangeOperation,
    ExpectedChange,
    TaskCategory,
    TaskCycle,
    TaskDifficulty,
    TaskSpec,
)


def make_task(
    task_id: str,
    *,
    split: DatasetSplit = DatasetSplit.TEST,
    arrival_minutes: int = 0,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        fixture_id="fixture-001",
        fixture_family_id="fixture-family-001",
        fixture_revision="fixture-sha-001",
        defect_family="cli-entry-point",
        split=split,
        category=TaskCategory.SINGLE_FILE_CHANGE,
        difficulty=TaskDifficulty.EASY,
        task="Correct the CLI entry point without changing unrelated files.",
        knowledge_distance=0.25,
        logical_arrival=timedelta(minutes=arrival_minutes),
        time_limit=timedelta(minutes=5),
        allowed_tools=("exec_command", "apply_patch"),
        expected_changes=(
            ExpectedChange(
                path="src/capability_capsule/cli/__init__.py",
                operation=ChangeOperation.MODIFY,
            ),
        ),
        validation_ids=("pytest:test_cli_entry_point",),
    )


def test_task_spec_round_trips_and_exposes_split_group() -> None:
    task = make_task("task-001")

    restored = TaskSpec.model_validate_json(task.model_dump_json())

    assert restored == task
    assert restored.schema_version == "0.1"
    assert restored.split_group == "fixture-family-001:cli-entry-point"


def test_read_only_task_can_have_no_expected_changes() -> None:
    task = TaskSpec(
        task_id="task-search-001",
        fixture_id="fixture-search",
        fixture_family_id="fixture-family-search",
        fixture_revision="fixture-sha-search",
        defect_family="code-location",
        split=DatasetSplit.TRAIN,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Find the CLI entry point and cite its path.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("exact-path:cli-entry-point",),
    )

    assert task.expected_changes == ()


def test_task_spec_rejects_non_positive_time_limit() -> None:
    payload = make_task("task-001").model_dump()
    payload["time_limit"] = timedelta(0)

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(payload)


def test_task_spec_rejects_blank_fixture_family_id() -> None:
    payload = make_task("task-001").model_dump()
    payload["fixture_family_id"] = " "

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.py", "src/../../outside.py", ""],
)
def test_expected_change_requires_safe_relative_path(path: str) -> None:
    with pytest.raises(ValidationError):
        ExpectedChange(path=path, operation=ChangeOperation.MODIFY)


def test_task_cycle_preserves_order_and_round_trips() -> None:
    first = make_task("task-001", arrival_minutes=0)
    second = make_task("task-002", arrival_minutes=10)
    cycle = TaskCycle(
        cycle_id="cycle-001",
        split=DatasetSplit.TEST,
        tasks=(first, second),
    )

    restored = TaskCycle.model_validate_json(cycle.model_dump_json())

    assert restored == cycle
    assert [task.task_id for task in restored.tasks] == ["task-001", "task-002"]


def test_task_cycle_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError):
        TaskCycle(
            cycle_id="cycle-001",
            split=DatasetSplit.TEST,
            tasks=(make_task("task-001"), make_task("task-001")),
        )


def test_task_cycle_rejects_tasks_from_another_split() -> None:
    with pytest.raises(ValidationError):
        TaskCycle(
            cycle_id="cycle-001",
            split=DatasetSplit.TEST,
            tasks=(make_task("task-001", split=DatasetSplit.TRAIN),),
        )


def test_task_cycle_rejects_decreasing_logical_arrival() -> None:
    with pytest.raises(ValidationError):
        TaskCycle(
            cycle_id="cycle-001",
            split=DatasetSplit.TEST,
            tasks=(
                make_task("task-001", arrival_minutes=10),
                make_task("task-002", arrival_minutes=5),
            ),
        )
