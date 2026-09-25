"""Tests for pinned Student inputs used by Teacher collection."""

from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.student_target import (
    BaseModelRecommendationReference,
    EnvironmentPurpose,
    HardwareProfileReference,
    StudentModelRole,
    StudentTarget,
    verify_student_target,
)
from capability_capsule.eval.tasks import TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection import TeacherAssignment, render_teacher_prompt
from capability_capsule.eval.teacher_collection_plan import TeacherCollectionPlan


def _write_json(path: Path, payload: object) -> str:
    encoded = json.dumps(payload, indent=2).encode() + b"\n"
    path.write_bytes(encoded)
    return sha256(encoded).hexdigest()


def make_target(tmp_path: Path) -> StudentTarget:
    hardware_path = tmp_path / "hardware.json"
    hardware_sha256 = _write_json(
        hardware_path,
        {
            "schema_version": "0.2",
            "profile_id": "hardware-laptop-001",
            "environment": {
                "verification_status": "confirmed",
                "purposes": [
                    "inference",
                    "evaluation",
                    "benchmarking",
                ],
            },
        },
    )
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_sha256 = _write_json(
        recommendation_path,
        {
            "schema_version": "0.1",
            "recommendation_id": "base-model-001",
            "status": "provisional",
            "hardware_profile_id": "hardware-laptop-001",
            "primary": {
                "model_id": "example/student-2b",
                "revision": "revision-001",
                "quantization": "q4_k_m",
                "runtime": "llama.cpp-001",
                "roles": ["executor", "coder"],
            },
        },
    )
    return StudentTarget(
        hardware=HardwareProfileReference(
            profile_id="hardware-laptop-001",
            path=hardware_path,
            sha256=hardware_sha256,
            verification_status="confirmed",
            purposes=(
                EnvironmentPurpose.INFERENCE,
                EnvironmentPurpose.EVALUATION,
                EnvironmentPurpose.BENCHMARKING,
            ),
        ),
        recommendation=BaseModelRecommendationReference(
            recommendation_id="base-model-001",
            path=recommendation_path,
            sha256=recommendation_sha256,
            evidence_status="provisional",
            accepted=True,
            hardware_profile_id="hardware-laptop-001",
            model_id="example/student-2b",
            revision="revision-001",
            quantization="q4_k_m",
            runtime="llama.cpp-001",
            roles=(
                StudentModelRole.EXECUTOR,
                StudentModelRole.CODER,
            ),
        ),
    )


def make_assignment(tmp_path: Path, target: StudentTarget | None) -> TeacherAssignment:
    task = TaskSpec(
        task_id="task-001",
        fixture_id="fixture-001",
        fixture_family_id="fixture-family-001",
        fixture_revision="fixture-revision-001",
        defect_family="entry-point-location",
        split=DatasetSplit.TRAIN,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Find the entry point.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("entry-point",),
    )
    return TeacherAssignment(
        assignment_id="assignment-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-6-astra",
        teacher_skill_version="0.2.0",
        authorized_fixture_root="fixtures/fixture-001",
        destination_jsonl=tmp_path / "teacher.jsonl",
        task=task,
        student_target=target,
    )


def test_verify_student_target_accepts_matching_immutable_artifacts(tmp_path: Path) -> None:
    verify_student_target(make_target(tmp_path))


def test_render_prompt_verifies_and_includes_student_target(tmp_path: Path) -> None:
    assignment = make_assignment(tmp_path, make_target(tmp_path))

    prompt = render_teacher_prompt(assignment)

    assert '"profile_id": "hardware-laptop-001"' in prompt
    assert '"recommendation_id": "base-model-001"' in prompt
    assert '"model_id": "example/student-2b"' in prompt
    assert '"roles": [' in prompt
    assert '"executor"' in prompt
    assert '"coder"' in prompt


def test_render_prompt_rejects_tampered_recommendation(tmp_path: Path) -> None:
    assignment = make_assignment(tmp_path, make_target(tmp_path))
    assert assignment.student_target is not None
    assignment.student_target.recommendation.path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        render_teacher_prompt(assignment)


def test_hardware_reference_rejects_unconfirmed_environment(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    payload = target.hardware.model_dump()
    payload["verification_status"] = "unverified"

    with pytest.raises(ValidationError, match="verification_status"):
        HardwareProfileReference.model_validate(payload)


def test_student_target_requires_inference_environment(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    hardware = target.hardware.model_copy(
        update={"purposes": (EnvironmentPurpose.EVALUATION,)},
    )

    with pytest.raises(ValidationError, match="inference purpose"):
        StudentTarget(
            hardware=hardware,
            recommendation=target.recommendation,
        )


def test_hardware_reference_rejects_duplicate_purposes(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    payload = target.hardware.model_dump()
    payload["purposes"] = ["inference", "inference"]

    with pytest.raises(ValidationError, match="purposes must be unique"):
        HardwareProfileReference.model_validate(payload)


def test_recommendation_rejects_duplicate_model_roles(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    payload = target.recommendation.model_dump()
    payload["roles"] = ["coder", "coder"]

    with pytest.raises(ValidationError, match="roles must be unique"):
        BaseModelRecommendationReference.model_validate(payload)


def test_plan_requires_every_assignment_to_use_its_student_target(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    assignment = make_assignment(tmp_path, None)

    with pytest.raises(ValidationError, match="Every assignment"):
        TeacherCollectionPlan(
            plan_id="plan-001",
            assignments=(assignment,),
            student_target=target,
        )


def test_plan_round_trip_preserves_one_target_for_every_assignment(tmp_path: Path) -> None:
    target = make_target(tmp_path)
    assignment = make_assignment(tmp_path, target)
    plan = TeacherCollectionPlan(
        plan_id="plan-001",
        assignments=(assignment,),
        student_target=target,
    )

    restored = TeacherCollectionPlan.model_validate_json(plan.model_dump_json())

    assert restored.schema_version == "0.2"
    assert restored.student_target == target
    assert restored.assignments[0].student_target == target


def test_schema_01_assignment_remains_loadable_without_student_target(tmp_path: Path) -> None:
    current = make_assignment(tmp_path, None)
    payload = current.model_dump(mode="json")
    payload["schema_version"] = "0.1"

    restored = TeacherAssignment.model_validate(payload)

    assert restored.schema_version == "0.1"
    assert restored.student_target is None
