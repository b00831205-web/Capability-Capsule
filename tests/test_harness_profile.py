"""Executable contract for immutable coding-harness profiles."""

from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest

from capability_capsule.eval.harness_profile import (
    HarnessProfileReference,
    verify_harness_profile,
)
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
from capability_capsule.eval.teacher_collection_plan import (
    TeacherCollectionPlan,
    pending_teacher_assignments,
)
from capability_capsule.eval.teacher_collection_plan_publication import (
    load_teacher_collection_plan,
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


def test_render_teacher_prompt_embeds_verified_harness_contract(
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

    if "Verified HarnessProfile" not in prompt:
        pytest.skip("Teacher prompt does not embed the verified HarnessProfile yet")

    assert '"arguments_schema"' in prompt
    assert '"required": [' in prompt
    assert '"cmd"' in prompt
    assert '"additionalProperties": false' in prompt
    assert '"result_envelope": "codex-exec-command-v1"' in prompt
    assert '"shell": "windows-powershell"' in prompt
    assert '"supports_multi_turn_tools": true' in prompt


def test_checked_in_codex_profile_and_schema_03_plan_are_reproducible() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    snapshot_path = (
        repository_root
        / "plans"
        / "harness"
        / "codex-0.154.0-alpha.6.2"
        / "contract-snapshot.json"
    )
    profile_path = snapshot_path.with_name("harness-profile.json")
    plan_path = (
        repository_root
        / "plans"
        / "teacher"
        / "smoke-pilot-003"
        / "collection-plan.json"
    )
    profile_payload = profile_path.read_bytes()
    profile_reference = HarnessProfileReference(
        profile_id="codex-0.154.0-alpha.6.2-windows-powershell-exec-v1",
        harness_id="codex",
        harness_version="0.154.0-alpha.6.2",
        path=profile_path,
        sha256=sha256(profile_payload).hexdigest(),
    )

    profile = verify_harness_profile(profile_reference)
    plan = TeacherCollectionPlan.model_validate_json(plan_path.read_text())
    prompt = render_teacher_prompt(
        plan.assignments[0],
        artifact_root=repository_root,
    )

    assert profile.prompt_template_sha256 == sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    assert profile.fixed_context_tokens == 9916
    assert plan.schema_version == "0.3"
    assert plan.harness_profile == plan.assignments[0].harness_profile
    assert plan.harness_profile is not None
    assert plan.harness_profile.sha256 == sha256(profile_payload).hexdigest()
    assert "Verified HarnessProfile" in prompt
    assert '"result_envelope": "codex-exec-command-v1"' in prompt


def test_checked_in_stage1_plan_has_required_split_counts_and_contract() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    published = load_teacher_collection_plan(
        repository_root / "plans" / "teacher" / "stage1-codex-001"
    )
    plan = published.plan
    train = tuple(
        assignment
        for assignment in plan.assignments
        if assignment.task.split is DatasetSplit.TRAIN
    )
    validation = tuple(
        assignment
        for assignment in plan.assignments
        if assignment.task.split is DatasetSplit.VALIDATION
    )

    assert published.schema_version == "0.3"
    assert plan.schema_version == "0.3"
    assert len(plan.assignments) == 10
    assert len(train) == 8
    assert len(validation) == 2
    assert all(
        assignment.harness_profile == plan.harness_profile
        for assignment in plan.assignments
    )
    assert all(
        assignment.student_target == plan.student_target
        for assignment in plan.assignments
    )
    assert all(
        assignment.task.split is not DatasetSplit.TEST
        for assignment in plan.assignments
    )
    assert pending_teacher_assignments(
        plan,
        artifact_root=repository_root,
    ) == ()
