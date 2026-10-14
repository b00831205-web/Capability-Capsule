"""Functional tests for validating Teacher trajectories against task specs."""

import inspect
import json
from datetime import timedelta
from typing import Any

import pytest

import capability_capsule.eval.dataset_validation as dataset_validation_module
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.harness_profile import HarnessProfile
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
    allowed_tools: tuple[str, ...] = ("exec_command",),
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
        allowed_tools=allowed_tools,
        validation_ids=("exact-path:cli-entry-point",),
    )


def make_trajectory(
    trajectory_id: str,
    *,
    task_id: str,
    split: DatasetSplit = DatasetSplit.TRAIN,
    source_revision: str = "fixture-sha-001",
    tool_name: str = "exec_command",
    tool_arguments: dict[str, Any] | None = None,
    tool_content: str = "src/capability_capsule/cli/__init__.py",
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
                        arguments=(
                            {"cmd": "rg entry"}
                            if tool_arguments is None
                            else tool_arguments
                        ),
                    ),
                ),
            ),
            TrajectoryMessage(
                role=MessageRole.TOOL,
                content=tool_content,
                tool_name=tool_name,
            ),
        ),
        source_revision=source_revision,
    )


def make_harness_profile() -> HarnessProfile:
    return HarnessProfile.model_validate(
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
        }
    )


def require_dataset_harness_support() -> None:
    parameters = inspect.signature(validate_teacher_dataset).parameters
    if "harness_profile" not in parameters:
        pytest.skip("Dataset validation does not support HarnessProfile yet")


def require_result_envelope_support() -> None:
    if not hasattr(
        dataset_validation_module,
        "_validate_tool_result_envelope",
    ):
        pytest.skip("Dataset validation does not verify tool result envelopes yet")


def codex_exec_result(
    *,
    envelope: str = "codex-exec-command-v1",
    exit_code: object = 0,
    output: object = "src/capability_capsule/cli/__init__.py",
) -> str:
    return json.dumps(
        {
            "envelope": envelope,
            "exit_code": exit_code,
            "output": output,
        },
        sort_keys=True,
    )


def test_teacher_dataset_accepts_tool_call_matching_harness_profile() -> None:
    require_dataset_harness_support()
    task = make_task("task-001")
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_content=codex_exec_result(),
    )

    validate_teacher_dataset(
        (trajectory,),
        tasks=(task,),
        harness_profile=make_harness_profile(),
    )


@pytest.mark.parametrize(
    "tool_arguments",
    [
        {},
        {"cmd": "rg entry", "unexpected": True},
    ],
)
def test_teacher_dataset_rejects_tool_arguments_outside_harness_schema(
    tool_arguments: dict[str, Any],
) -> None:
    require_dataset_harness_support()
    task = make_task("task-001")
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_arguments=tool_arguments,
    )

    with pytest.raises(ValueError, match="exec_command"):
        validate_teacher_dataset(
            (trajectory,),
            tasks=(task,),
            harness_profile=make_harness_profile(),
        )


def test_teacher_dataset_rejects_task_tool_missing_from_harness_profile() -> None:
    require_dataset_harness_support()
    task = make_task(
        "task-001",
        allowed_tools=("exec_command", "web_search"),
    )
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_name="web_search",
        tool_arguments={"query": "entry point"},
    )

    with pytest.raises(ValueError, match="web_search"):
        validate_teacher_dataset(
            (trajectory,),
            tasks=(task,),
            harness_profile=make_harness_profile(),
        )


def test_teacher_dataset_accepts_matching_tool_result_envelope() -> None:
    require_result_envelope_support()
    task = make_task("task-001")
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_content=codex_exec_result(),
    )

    validate_teacher_dataset(
        (trajectory,),
        tasks=(task,),
        harness_profile=make_harness_profile(),
    )


@pytest.mark.parametrize(
    "tool_content",
    [
        "plain unstructured output",
        codex_exec_result(envelope="different-envelope"),
        codex_exec_result(exit_code="0"),
        codex_exec_result(output=123),
    ],
)
def test_teacher_dataset_rejects_invalid_tool_result_envelope(
    tool_content: str,
) -> None:
    require_result_envelope_support()
    task = make_task("task-001")
    trajectory = make_trajectory(
        "trajectory-001",
        task_id="task-001",
        tool_content=tool_content,
    )

    with pytest.raises(ValueError, match="codex-exec-command-v1"):
        validate_teacher_dataset(
            (trajectory,),
            tasks=(task,),
            harness_profile=make_harness_profile(),
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
