"""Executable contract for immutable coding-harness profiles."""

from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest

from capability_capsule.eval.harness_profile import HarnessProfileReference
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.tasks import (
    TaskCategory,
    TaskDifficulty,
    TaskSpec,
)
from capability_capsule.eval.teacher_collection import (
    TeacherAssignment,
    render_teacher_prompt,
)


def _write_profile(path: Path) -> str:
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
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def _harness_profile_api() -> tuple[object, object]:
    module = import_module("capability_capsule.eval.harness_profile")
    return module.HarnessProfileReference, module.verify_harness_profile


def test_harness_profile_reference_verifies_pinned_contract(tmp_path: Path) -> None:
    reference_type, verify = _harness_profile_api()
    profile_path = tmp_path / "codex-profile.json"
    digest = _write_profile(profile_path)

    reference = reference_type(
        profile_id="codex-windows-powershell-001",
        harness_id="codex",
        harness_version="pinned-version-001",
        path=profile_path,
        sha256=digest,
    )

    verified = verify(reference)

    assert verified.profile_id == "codex-windows-powershell-001"
    assert verified.tools[0].name == "exec_command"
    assert verified.tools[0].arguments_schema["required"] == ["cmd"]
    assert verified.tools[0].result_envelope == "codex-exec-command-v1"


def test_harness_profile_reference_rejects_tampered_artifact(tmp_path: Path) -> None:
    reference_type, verify = _harness_profile_api()
    profile_path = tmp_path / "codex-profile.json"
    digest = _write_profile(profile_path)
    reference = reference_type(
        profile_id="codex-windows-powershell-001",
        harness_id="codex",
        harness_version="pinned-version-001",
        path=profile_path,
        sha256=digest,
    )
    profile_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(reference)


def _make_harness_reference(tmp_path: Path) -> HarnessProfileReference:
    profile_path = tmp_path / "codex-profile.json"
    digest = _write_profile(profile_path)
    return HarnessProfileReference(
        profile_id="codex-windows-powershell-001",
        harness_id="codex",
        harness_version="pinned-version-001",
        path=profile_path,
        sha256=digest,
    )


def _make_task() -> TaskSpec:
    return TaskSpec(
        task_id="task-001",
        fixture_id="fixture-001",
        fixture_family_id="fixture-family-001",
        fixture_revision="fixture-revision-001",
        defect_family="single-file-edit",
        split=DatasetSplit.TRAIN,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Fix the greeting.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("greeting-validator",),
    )


def _require_assignment_harness_support() -> None:
    if "harness_profile" not in TeacherAssignment.model_fields:
        pytest.skip("TeacherAssignment does not support HarnessProfile yet")


def test_render_teacher_prompt_verifies_and_includes_harness_profile(
    tmp_path: Path,
) -> None:
    _require_assignment_harness_support()
    reference = _make_harness_reference(tmp_path)
    assignment = TeacherAssignment(
        schema_version="0.3",
        assignment_id="assignment-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        authorized_fixture_root="/fixtures/greeting",
        destination_jsonl=tmp_path / "teacher.jsonl",
        task=_make_task(),
        harness_profile=reference,
    )

    prompt = render_teacher_prompt(assignment)

    assert '"schema_version": "0.3"' in prompt
    assert '"profile_id": "codex-windows-powershell-001"' in prompt
    assert '"harness_id": "codex"' in prompt
    assert '"harness_version": "pinned-version-001"' in prompt


def test_render_teacher_prompt_rejects_tampered_harness_profile(
    tmp_path: Path,
) -> None:
    _require_assignment_harness_support()
    reference = _make_harness_reference(tmp_path)
    assignment = TeacherAssignment(
        schema_version="0.3",
        assignment_id="assignment-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        authorized_fixture_root="/fixtures/greeting",
        destination_jsonl=tmp_path / "teacher.jsonl",
        task=_make_task(),
        harness_profile=reference,
    )
    reference.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        render_teacher_prompt(assignment)
