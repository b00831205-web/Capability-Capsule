"""Functional tests for reproducible batch Teacher collection plans."""

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.records import DatasetSplit, MessageRole, TeacherTrajectory, TrajectoryMessage
from capability_capsule.eval.tasks import TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection import append_teacher_trajectory
from capability_capsule.eval.teacher_collection_plan import (
    TeacherCollectionPlan,
    build_teacher_collection_plan,
    pending_teacher_assignments,
)


def make_task(
    task_id: str,
    fixture_id: str,
    split: DatasetSplit,
    *,
    defect_family: str = "entry-point-location",
    fixture_family_id: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        fixture_id=fixture_id,
        fixture_family_id=fixture_family_id or f"family-{fixture_id}",
        fixture_revision=f"revision-{fixture_id}",
        defect_family=defect_family,
        split=split,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task=f"Complete {task_id}.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=(f"validator-{task_id}",),
    )


def make_trajectory(assignment_index: int, plan: TeacherCollectionPlan) -> TeacherTrajectory:
    assignment = plan.assignments[assignment_index]
    task = assignment.task
    return TeacherTrajectory(
        trajectory_id=assignment.trajectory_id,
        task_id=task.task_id,
        split=task.split,
        teacher_model=assignment.teacher_model,
        teacher_skill_version=assignment.teacher_skill_version,
        task=task.task,
        messages=(
            TrajectoryMessage(role=MessageRole.USER, content=task.task),
            TrajectoryMessage(role=MessageRole.ASSISTANT, content="Completed."),
        ),
        source_revision=task.fixture_revision,
    )


def build_plan(tmp_path: Path) -> TeacherCollectionPlan:
    tasks = (
        make_task("task-train", "fixture-train", DatasetSplit.TRAIN),
        make_task(
            "task-validation",
            "fixture-validation",
            DatasetSplit.VALIDATION,
        ),
    )
    return build_teacher_collection_plan(
        plan_id="pilot-001",
        tasks=tasks,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        fixture_roots={
            "fixture-train": "/fixtures/train",
            "fixture-validation": "/fixtures/validation",
        },
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        trajectory_ids={
            "task-train": "trajectory-train-001",
            "task-validation": "trajectory-validation-001",
        },
    )


def test_build_teacher_collection_plan_creates_ordered_assignments(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)

    assert plan.plan_id == "pilot-001"
    assert [assignment.assignment_id for assignment in plan.assignments] == [
        "pilot-001:task-train",
        "pilot-001:task-validation",
    ]
    assert [assignment.trajectory_id for assignment in plan.assignments] == [
        "trajectory-train-001",
        "trajectory-validation-001",
    ]
    assert plan.assignments[0].authorized_fixture_root == "/fixtures/train"
    assert plan.assignments[1].authorized_fixture_root == "/fixtures/validation"


def test_build_teacher_collection_plan_requires_fixture_provenance(tmp_path: Path) -> None:
    task = make_task("task-001", "fixture-missing", DatasetSplit.TRAIN)

    with pytest.raises(ValueError, match=r"Missing fixture root.*fixture-missing"):
        build_teacher_collection_plan(
            plan_id="pilot-001",
            tasks=(task,),
            teacher_model="gpt-6-astra",
            teacher_skill_version="0.1.0",
            fixture_roots={},
            destination_jsonl=tmp_path / "teacher.jsonl",
            trajectory_ids={"task-001": "trajectory-001"},
        )


def test_build_teacher_collection_plan_requires_explicit_trajectory_ids(tmp_path: Path) -> None:
    task = make_task("task-001", "fixture-001", DatasetSplit.TRAIN)

    with pytest.raises(ValueError, match=r"Missing trajectory ID.*task-001"):
        build_teacher_collection_plan(
            plan_id="pilot-001",
            tasks=(task,),
            teacher_model="gpt-6-astra",
            teacher_skill_version="0.1.0",
            fixture_roots={"fixture-001": "/fixtures/001"},
            destination_jsonl=tmp_path / "teacher.jsonl",
            trajectory_ids={},
        )


def test_collection_plan_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    duplicate = plan.assignments[0].model_copy(
        update={"assignment_id": plan.assignments[1].assignment_id}
    )

    with pytest.raises(ValidationError, match="Duplicate assignment ID"):
        TeacherCollectionPlan(
            plan_id="duplicate-plan",
            assignments=(plan.assignments[1], duplicate),
        )


def test_collection_plan_rejects_split_group_leakage(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    validation_task = make_task(
        "task-leak",
        "fixture-renamed-copy",
        DatasetSplit.VALIDATION,
        fixture_family_id="family-fixture-train",
    )
    leaking_assignment = plan.assignments[1].model_copy(
        update={
            "assignment_id": "assignment-leak",
            "trajectory_id": "trajectory-leak",
            "task": validation_task,
        }
    )

    with pytest.raises(ValidationError, match="Split leakage"):
        TeacherCollectionPlan(
            plan_id="leaking-plan",
            assignments=(plan.assignments[0], leaking_assignment),
        )


def test_pending_teacher_assignments_supports_resuming(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)
    completed_assignment = plan.assignments[0]
    append_teacher_trajectory(
        completed_assignment,
        make_trajectory(0, plan),
    )

    pending = pending_teacher_assignments(plan)

    assert pending == (plan.assignments[1],)


def test_pending_teacher_assignments_returns_entire_new_plan(tmp_path: Path) -> None:
    plan = build_plan(tmp_path)

    assert pending_teacher_assignments(plan) == plan.assignments
