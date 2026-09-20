"""Functional tests for reproducible batch Teacher collection plans."""

import inspect
import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.harness_profile import HarnessProfileReference
from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
)
from capability_capsule.eval.tasks import TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection import append_teacher_trajectory
from capability_capsule.eval.teacher_collection_plan import (
    TeacherCollectionPlan,
    build_teacher_collection_plan,
    pending_teacher_assignments,
)
from capability_capsule.eval.teacher_collection_plan_publication import (
    load_teacher_collection_plan,
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


def make_harness_profile(tmp_path: Path) -> HarnessProfileReference:
    profile_path = tmp_path / "codex-profile.json"
    encoded = json.dumps(
        {
            "schema_version": "0.1",
            "profile_id": "codex-windows-powershell-001",
            "harness_id": "codex",
            "harness_version": "pinned-version-001",
            "provider_protocol": "openai-compatible-chat-completions",
            "model_injection": "invocation-scoped-config",
            "prompt_template_sha256": "1" * 64,
            "fixed_context_tokens": 12000,
            "shell": "windows-powershell",
            "filesystem": "workspace-write",
            "approval_policy": "on-request",
            "sandbox_policy": "workspace-write",
            "supports_parallel_tools": False,
            "supports_multi_turn_tools": True,
            "tools": [
                {
                    "name": "exec_command",
                    "arguments_schema": {
                        "type": "object",
                        "required": ["cmd"],
                        "additionalProperties": False,
                        "properties": {"cmd": {"type": "string"}},
                    },
                    "result_envelope": "codex-exec-command-v1",
                }
            ],
            "validator_ids": ["codex-tool-envelope-v1"],
        },
        sort_keys=True,
    ).encode() + b"\n"
    profile_path.write_bytes(encoded)
    return HarnessProfileReference(
        profile_id="codex-windows-powershell-001",
        harness_id="codex",
        harness_version="pinned-version-001",
        path=profile_path,
        sha256=sha256(encoded).hexdigest(),
    )


def require_plan_harness_support() -> None:
    builder_parameters = inspect.signature(build_teacher_collection_plan).parameters
    if (
        "harness_profile" not in TeacherCollectionPlan.model_fields
        or "harness_profile" not in builder_parameters
    ):
        pytest.skip("TeacherCollectionPlan does not support HarnessProfile yet")


def require_pending_harness_support() -> None:
    parameters = inspect.signature(pending_teacher_assignments).parameters
    if "artifact_root" not in parameters:
        pytest.skip("Pending-assignment validation does not verify HarnessProfile yet")


def build_harness_plan(tmp_path: Path) -> TeacherCollectionPlan:
    task = make_task("task-train", "fixture-train", DatasetSplit.TRAIN)
    return build_teacher_collection_plan(
        plan_id="pilot-harness-001",
        tasks=(task,),
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        fixture_roots={"fixture-train": "/fixtures/train"},
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        trajectory_ids={"task-train": "trajectory-train-001"},
        harness_profile=make_harness_profile(tmp_path),
    )


def test_build_plan_propagates_one_harness_profile_to_every_assignment(
    tmp_path: Path,
) -> None:
    require_plan_harness_support()
    profile = make_harness_profile(tmp_path)
    tasks = (
        make_task("task-train", "fixture-train", DatasetSplit.TRAIN),
        make_task(
            "task-validation",
            "fixture-validation",
            DatasetSplit.VALIDATION,
        ),
    )

    plan = build_teacher_collection_plan(
        plan_id="pilot-harness-001",
        tasks=tasks,
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        fixture_roots={
            "fixture-train": "/fixtures/train",
            "fixture-validation": "/fixtures/validation",
        },
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        trajectory_ids={
            "task-train": "trajectory-train-001",
            "task-validation": "trajectory-validation-001",
        },
        harness_profile=profile,
    )

    assert plan.schema_version == "0.3"
    assert plan.harness_profile == profile
    assert all(
        assignment.schema_version == "0.3"
        and assignment.harness_profile == profile
        for assignment in plan.assignments
    )


def test_plan_rejects_assignment_with_a_different_harness_profile(
    tmp_path: Path,
) -> None:
    require_plan_harness_support()
    profile = make_harness_profile(tmp_path)
    task = make_task("task-train", "fixture-train", DatasetSplit.TRAIN)
    plan = build_teacher_collection_plan(
        plan_id="pilot-harness-001",
        tasks=(task,),
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        fixture_roots={"fixture-train": "/fixtures/train"},
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        trajectory_ids={"task-train": "trajectory-train-001"},
        harness_profile=profile,
    )
    other_profile = profile.model_copy(
        update={"profile_id": "codex-windows-powershell-002"}
    )
    mismatched_assignment = plan.assignments[0].model_copy(
        update={"harness_profile": other_profile}
    )

    with pytest.raises(ValidationError, match="HarnessProfile"):
        TeacherCollectionPlan(
            schema_version="0.3",
            plan_id="pilot-harness-mismatch",
            assignments=(mismatched_assignment,),
            harness_profile=profile,
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


def test_pending_assignments_rejects_tampered_harness_profile(
    tmp_path: Path,
) -> None:
    require_pending_harness_support()
    plan = build_harness_plan(tmp_path)
    append_teacher_trajectory(
        plan.assignments[0],
        make_trajectory(0, plan),
        artifact_root=tmp_path,
    )
    assert plan.harness_profile is not None
    plan.harness_profile.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pending_teacher_assignments(plan, artifact_root=tmp_path)


def test_pending_assignments_revalidates_existing_tool_calls(
    tmp_path: Path,
) -> None:
    require_pending_harness_support()
    plan = build_harness_plan(tmp_path)
    trajectory = make_trajectory(0, plan).model_copy(
        update={
            "messages": (
                TrajectoryMessage(
                    role=MessageRole.USER,
                    content="Complete task-train.",
                ),
                TrajectoryMessage(
                    role=MessageRole.ASSISTANT,
                    content="I will inspect the project.",
                    tool_calls=(
                        ToolCallRecord(
                            name="exec_command",
                            arguments={},
                        ),
                    ),
                ),
                TrajectoryMessage(
                    role=MessageRole.TOOL,
                    content="No output.",
                    tool_name="exec_command",
                ),
            )
        }
    )
    append_jsonl(
        plan.assignments[0].destination_jsonl,
        trajectory,
    )

    with pytest.raises(ValueError, match="exec_command"):
        pending_teacher_assignments(plan, artifact_root=tmp_path)


def test_checked_in_stage1_powershell_collection_is_complete() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    published = load_teacher_collection_plan(
        repository_root
        / "plans"
        / "teacher"
        / "stage1-codex-powershell-002"
    )
    raw_path = (
        repository_root
        / "datasets"
        / "teacher"
        / "stage1-codex-powershell-002"
        / "raw.jsonl"
    )
    trajectories = load_jsonl(raw_path, TeacherTrajectory)

    assert len(published.plan.assignments) == 8
    assert len(trajectories) == 8
    assert {trajectory.split for trajectory in trajectories} == {
        DatasetSplit.TRAIN
    }
    assert all(
        "powershell-native" in trajectory.tags
        for trajectory in trajectories
    )
    assert pending_teacher_assignments(
        published.plan,
        artifact_root=repository_root,
    ) == ()
