"""Cross-record validation for Teacher trajectory datasets."""

from collections.abc import Sequence

from capability_capsule.eval.curation import validate_split_isolation
from capability_capsule.eval.records import(
    DatasetSplit,
    TeacherTrajectory,
)
from typing import Any
import json

from capability_capsule.eval.harness_profile import HarnessProfile, HarnessToolProfile

from capability_capsule.eval.tasks import TaskSpec


def validate_teacher_dataset(
        trajectories: Sequence[TeacherTrajectory],
        *,
        tasks: Sequence[TaskSpec],
        harness_profile: HarnessProfile | None = None,
) -> None:
    """Validate Teacher trajectories against their registered task specs"""

    task_by_id: dict[str, TaskSpec] = {}

    for task in tasks:
        if task.task_id in task_by_id:
            raise ValueError(f"Duplicate task ID {task.task_id!r}")

        task_by_id[task.task_id] = task

    seen_trajectory_ids: set[str] = set()
    group_by_task: dict[str, str] = {}

    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ValueError(
                f"Duplicate trajectory ID {trajectory.trajectory_id!r}"
            )

        seen_trajectory_ids.add(trajectory.trajectory_id)

        if trajectory.split is DatasetSplit.TEST:
            raise ValueError(
                f"Teacher trajectory {trajectory.trajectory_id!r}"
                "must not use the locked test split"
            )

        try:
            task = task_by_id[trajectory.task_id]
        except KeyError as error:
            raise ValueError(
                f"Teacher trajectory reference unknown task "
                f"{trajectory.task_id!r}"
            ) from error

        if trajectory.split is not task.split:
            raise ValueError(
                f"Trajectory split does not match task "
                f"{trajectory.task_id!r}"
            )

        if trajectory.source_revision != task.fixture_revision:
            raise ValueError(
                f"Trajectory source revision does not match task "
                f"{trajectory.task_id!r}"
            )

        if trajectory.task != task.task:
            raise ValueError(
                f"Trajectory text does not match task "
                f"{trajectory.task_id!r}"
            )

        allowed_tools = set(task.allowed_tools)

        for message in trajectory.messages:
            for tool_call in message.tool_calls:
                if tool_call.name not in allowed_tools:
                    raise ValueError(
                        f"Tool {tool_call.name!r} is not allowed for "
                        f"task {trajectory.task_id!r}"
                    )
                if harness_profile is not None:
                    tool_profile = _tool_by_name(harness_profile, tool_call.name)
                    _validate_tool_arguments(tool_name = tool_call.name, arguments = tool_call.arguments, tool_profile= tool_profile)

            if (
                message.tool_name is not None
                and message.tool_name not in allowed_tools
            ):
                    raise ValueError(
                        f"Tool {message.tool_name!r} is not allowed for "
                        f"task {trajectory.task_id!r}"
                    )

            if harness_profile is not None and message.tool_name is not None:
                tool_profile = _tool_by_name(harness_profile, message.tool_name)
                _validate_tool_result_envelope(tool_name=message.tool_name, content=message.content, tool_profile=tool_profile)
        group_by_task[trajectory.task_id] = task.split_group

    validate_split_isolation(
        trajectories,
        group_by_task=group_by_task
    )

def _tool_by_name(profile: HarnessToolProfile, tool_name: str) -> HarnessToolProfile:
    for tool in profile.tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(
        f"Tool {tool_name!r} is not defined by HarnessProfile {profile.profile_id!r}"
    )

def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type == "object":
        return isinstance(value, dict)

    if expected_type == "null":
        return value is None

    if expected_type == "array":
        return isinstance(value, list)

    raise ValueError(f"Unsupported HarnessProfile JSON type {expected_type!r}")        

def _validate_tool_arguments(*, tool_name: str, arguments: dict[str, Any], tool_profile: HarnessToolProfile) -> None:
    schema = tool_profile.arguments_schema
    properties = schema["properties"]
    required = schema.get("required", [])

    missing = [
        name for name in required if name not in arguments
    ]
    if missing:
        raise ValueError(
            f"Tool {tool_name!r} is missing required arguments: {', '.join(sorted(missing))}"
        )

    if schema.get("additionalProperties") is False:
        unexpected = set(arguments) - set(properties)
        if unexpected:
            raise ValueError(
                f"Tool {tool_name!r} contains unexpected arguments: {', '.join(sorted(unexpected))}"
            )

    for argument_name, argument_value in arguments.items():
        property_schema = properties.get(argument_name)
        if property_schema is None:
            continue

        if not isinstance(property_schema, dict):
            raise ValueError(
                f"Tool {tool_name!r} argument schema for {argument_name!r} must be an object"
            )

        expected_type = property_schema.get("type")
        if expected_type is None:
            continue

        if not isinstance(expected_type, str):
            raise ValueError(
                f"Tool {tool_name!r} argument type for {argument_name!r} must be a string"
            )

        if not _matches_json_type(argument_value, expected_type):
            raise ValueError(
                f"Tool {tool_name!r} argument {argument_name!r} must have JSON type {expected_type!r}"
            )

def _validate_tool_result_envelope(
        *,
        tool_name: str,
        content: str,
        tool_profile: HarnessProfile,
) -> None:

    envelope_id = tool_profile.result_envelope

    if envelope_id != "codex-exec-command-v1":
        raise ValueError(
            f"Unsupported tool result envelope {envelope_id!r} for tool {tool_name!r}"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Tool {tool_name!r} result must use {envelope_id!r} JSON"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"Tool {tool_name!r} result must use {envelope_id!r} JSON object"
        )

    required_fields = {"envelope", "exit_code", "output"}
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise ValueError(
            f"Tool {tool_name!r} result using {envelope_id!r} is missing fields: {', '.join(sorted(missing_fields))}"
        )

    if payload["envelope"] != envelope_id:
        raise ValueError(
            f"Tool {tool_name!r} result must identify {envelope_id!r}"
        )

    unexpected_fields = set(payload) - required_fields
    if unexpected_fields:
        raise ValueError(
            f"Tool {tool_name!r} result using {envelope_id!r} "
            f"contains unexpected fields: {', '.join(sorted(unexpected_fields))}"
        )

    exit_code = payload["exit_code"]
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError(
            f"Tool {tool_name!r} result using {envelope_id!r} must contain an integer exit_code"
        )

    if not isinstance(payload["output"], str):
        raise ValueError(
            f"Tool {tool_name!r} result using {envelope_id!r} must contain a string output"
        )