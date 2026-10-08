"""Functional tests for preparing and collecting Teacher trajectories."""

import inspect
import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.harness_profile import HarnessProfileReference
from capability_capsule.eval.jsonl import load_jsonl
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
from capability_capsule.eval.teacher_collection import (
    TeacherAssignment,
    append_teacher_trajectory,
    render_teacher_prompt,
)


def make_task(split: DatasetSplit = DatasetSplit.TRAIN) -> TaskSpec:
    return TaskSpec(
        task_id="task-001",
        fixture_id="fixture-cli",
        fixture_family_id="fixture-family-cli",
        fixture_revision="fixture-sha-001",
        defect_family="entry-point-location",
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


def make_assignment(tmp_path: Path) -> TeacherAssignment:
    return TeacherAssignment(
        assignment_id="assignment-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        authorized_fixture_root="/fixtures/fixture-cli",
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        task=make_task(),
    )


def make_harness_profile(tmp_path: Path) -> HarnessProfileReference:
    profile_path = tmp_path / "profiles" / "codex.json"
    profile_path.parent.mkdir(parents=True)
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
        path=Path("profiles/codex.json"),
        sha256=sha256(encoded).hexdigest(),
    )


def make_harness_assignment(tmp_path: Path) -> TeacherAssignment:
    return TeacherAssignment(
        schema_version="0.3",
        assignment_id="assignment-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.1.0",
        authorized_fixture_root="/fixtures/fixture-cli",
        destination_jsonl=tmp_path / "raw" / "teacher.jsonl",
        task=make_task(),
        harness_profile=make_harness_profile(tmp_path),
    )


def require_append_harness_support() -> None:
    parameters = inspect.signature(append_teacher_trajectory).parameters
    if "artifact_root" not in parameters:
        pytest.skip("Teacher append does not verify HarnessProfile yet")


def make_trajectory(
    *,
    trajectory_id: str = "trajectory-001",
    teacher_model: str = "gpt-6-astra",
    teacher_skill_version: str = "0.1.0",
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id="task-001",
        split=DatasetSplit.TRAIN,
        teacher_model=teacher_model,
        teacher_skill_version=teacher_skill_version,
        task="Find the CLI entry point.",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Find the CLI entry point.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="The entry point is capability_capsule.cli:app.",
            ),
        ),
        source_revision="fixture-sha-001",
    )


def test_render_teacher_prompt_contains_complete_auditable_assignment(
    tmp_path: Path,
) -> None:
    assignment = make_assignment(tmp_path)

    prompt = render_teacher_prompt(assignment)

    assert prompt.startswith("Use $capsule-teacher")
    assert '"assignment_id": "assignment-001"' in prompt
    assert '"trajectory_id": "trajectory-001"' in prompt
    assert '"fixture_revision": "fixture-sha-001"' in prompt
    assert '"allowed_tools": [' in prompt
    assert '"exec_command"' in prompt
    assert '"validation_ids": [' in prompt
    assert str(assignment.destination_jsonl) in prompt


def test_teacher_assignment_rejects_locked_test_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="locked test split"):
        TeacherAssignment(
            assignment_id="assignment-test",
            trajectory_id="trajectory-test",
            teacher_model="gpt-6-astra",
            teacher_skill_version="0.1.0",
            authorized_fixture_root="/fixtures/fixture-test",
            destination_jsonl=tmp_path / "teacher.jsonl",
            task=make_task(DatasetSplit.TEST),
        )


def test_append_teacher_trajectory_validates_and_appends(tmp_path: Path) -> None:
    assignment = make_assignment(tmp_path)
    trajectory = make_trajectory()

    append_teacher_trajectory(assignment, trajectory)

    assert load_jsonl(
        assignment.destination_jsonl,
        TeacherTrajectory,
    ) == (trajectory,)


def test_append_teacher_trajectory_rejects_duplicate_id_without_appending(
    tmp_path: Path,
) -> None:
    assignment = make_assignment(tmp_path)
    trajectory = make_trajectory()
    append_teacher_trajectory(assignment, trajectory)

    with pytest.raises(ValueError, match="already exists"):
        append_teacher_trajectory(assignment, trajectory)

    assert load_jsonl(
        assignment.destination_jsonl,
        TeacherTrajectory,
    ) == (trajectory,)


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        (make_trajectory(trajectory_id="wrong-id"), "Trajectory ID"),
        (make_trajectory(teacher_model="wrong-model"), "Teacher model"),
        (
            make_trajectory(teacher_skill_version="9.9.9"),
            "Teacher skill version",
        ),
    ],
)
def test_append_teacher_trajectory_rejects_assignment_mismatch(
    tmp_path: Path,
    trajectory: TeacherTrajectory,
    message: str,
) -> None:
    assignment = make_assignment(tmp_path)

    with pytest.raises(ValueError, match=message):
        append_teacher_trajectory(assignment, trajectory)

    assert not assignment.destination_jsonl.exists()


def test_append_teacher_trajectory_enforces_allowed_tools(tmp_path: Path) -> None:
    assignment = make_assignment(tmp_path)
    trajectory = make_trajectory().model_copy(
        update={
            "messages": (
                TrajectoryMessage(
                    role=MessageRole.USER,
                    content="Find the CLI entry point.",
                ),
                TrajectoryMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(
                        ToolCallRecord(name="forbidden_tool"),
                    ),
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        append_teacher_trajectory(assignment, trajectory)

    assert not assignment.destination_jsonl.exists()


def test_append_teacher_trajectory_rejects_tampered_harness_profile(
    tmp_path: Path,
) -> None:
    require_append_harness_support()
    assignment = make_harness_assignment(tmp_path)
    assert assignment.harness_profile is not None
    (tmp_path / assignment.harness_profile.path).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        append_teacher_trajectory(
            assignment,
            make_trajectory(),
            artifact_root=tmp_path,
        )

    assert not assignment.destination_jsonl.exists()


def test_append_teacher_trajectory_enforces_harness_tool_schema(
    tmp_path: Path,
) -> None:
    require_append_harness_support()
    assignment = make_harness_assignment(tmp_path)
    trajectory = make_trajectory().model_copy(
        update={
            "messages": (
                TrajectoryMessage(
                    role=MessageRole.USER,
                    content="Find the CLI entry point.",
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

    with pytest.raises(ValueError, match="exec_command"):
        append_teacher_trajectory(
            assignment,
            trajectory,
            artifact_root=tmp_path,
        )

    assert not assignment.destination_jsonl.exists()
