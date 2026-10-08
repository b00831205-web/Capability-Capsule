"""Functional tests for immutable Teacher collection-plan publication."""

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import get_args

import pytest

from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.tasks import TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection import TeacherAssignment
from capability_capsule.eval.teacher_collection_plan import TeacherCollectionPlan
from capability_capsule.eval.teacher_collection_plan_publication import (
    PublishedTeacherCollectionPlan,
    load_teacher_collection_plan,
    publish_teacher_collection_plan,
)


def make_plan(tmp_path: Path) -> TeacherCollectionPlan:
    task = TaskSpec(
        task_id="task-001",
        fixture_id="fixture-001",
        fixture_family_id="fixture-family-001",
        fixture_revision="fixture-sha-001",
        defect_family="entry-point-location",
        split=DatasetSplit.TRAIN,
        category=TaskCategory.CODE_SEARCH,
        difficulty=TaskDifficulty.EASY,
        task="Find the CLI entry point.",
        knowledge_distance=0.1,
        logical_arrival=timedelta(0),
        time_limit=timedelta(minutes=2),
        allowed_tools=("exec_command",),
        validation_ids=("exact-path:__main__.py",),
    )
    assignment = TeacherAssignment(
        assignment_id="pilot-001:task-001",
        trajectory_id="trajectory-001",
        teacher_model="gpt-5",
        teacher_skill_version="0.1.0",
        authorized_fixture_root="fixtures/pilot/fixture-001",
        destination_jsonl=Path("datasets/teacher/pilot/raw.jsonl"),
        task=task,
    )
    return TeacherCollectionPlan(
        plan_id="pilot-001",
        assignments=(assignment,),
    )


def require_schema_03_publication_support() -> None:
    annotation = PublishedTeacherCollectionPlan.model_fields[
        "schema_version"
    ].annotation
    if "0.3" not in get_args(annotation):
        pytest.skip("Collection-plan publication does not support schema 0.3 yet")


def load_checked_in_schema_03_plan() -> TeacherCollectionPlan:
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "plans"
        / "teacher"
        / "smoke-pilot-003"
        / "collection-plan.json"
    )
    return TeacherCollectionPlan.model_validate_json(plan_path.read_text())


def test_publish_teacher_collection_plan_writes_immutable_portable_record(
    tmp_path: Path,
) -> None:
    plan = make_plan(tmp_path)

    published = publish_teacher_collection_plan(
        plan,
        output_root=tmp_path,
    )

    plan_dir = tmp_path / "pilot-001"
    plan_path = plan_dir / "collection-plan.json"
    manifest_path = plan_dir / "manifest.json"

    assert published.plan == plan
    assert published.plan_id == "pilot-001"
    assert published.plan_filename == "collection-plan.json"
    assert published.byte_count == plan_path.stat().st_size
    assert published.sha256 == sha256(plan_path.read_bytes()).hexdigest()
    assert load_teacher_collection_plan(plan_dir) == published
    assert str(tmp_path) not in plan_path.read_text(encoding="utf-8")
    assert PublishedTeacherCollectionPlan.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    ) == published


def test_publish_teacher_collection_plan_refuses_overwrite(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    publish_teacher_collection_plan(plan, output_root=tmp_path)
    original = (tmp_path / "pilot-001" / "collection-plan.json").read_bytes()

    with pytest.raises(FileExistsError, match="pilot-001"):
        publish_teacher_collection_plan(plan, output_root=tmp_path)

    assert (tmp_path / "pilot-001" / "collection-plan.json").read_bytes() == original


@pytest.mark.parametrize("plan_id", ["", " ", ".", "..", "../escape", "nested/plan"])
def test_publish_teacher_collection_plan_rejects_unsafe_plan_id(
    tmp_path: Path,
    plan_id: str,
) -> None:
    plan = make_plan(tmp_path).model_copy(update={"plan_id": plan_id})

    with pytest.raises(ValueError, match="plan_id"):
        publish_teacher_collection_plan(plan, output_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_load_teacher_collection_plan_rejects_tampered_plan(
    tmp_path: Path,
) -> None:
    plan = make_plan(tmp_path)
    publish_teacher_collection_plan(plan, output_root=tmp_path)
    plan_path = tmp_path / "pilot-001" / "collection-plan.json"
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=r"collection-plan\.json.*SHA-256"):
        load_teacher_collection_plan(tmp_path / "pilot-001")


def test_load_teacher_collection_plan_rejects_unexpected_files(tmp_path: Path) -> None:
    plan = make_plan(tmp_path)
    publish_teacher_collection_plan(plan, output_root=tmp_path)
    (tmp_path / "pilot-001" / "notes.txt").write_text("untracked", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected.*notes\.txt"):
        load_teacher_collection_plan(tmp_path / "pilot-001")


def test_publish_schema_03_plan_preserves_harness_profile_and_manifest_version(
    tmp_path: Path,
) -> None:
    require_schema_03_publication_support()
    plan = load_checked_in_schema_03_plan()

    published = publish_teacher_collection_plan(
        plan,
        output_root=tmp_path,
    )
    plan_dir = tmp_path / "smoke-pilot-003"
    plan_path = plan_dir / "collection-plan.json"

    assert published.schema_version == "0.3"
    assert published.plan.schema_version == "0.3"
    assert published.plan.harness_profile == plan.harness_profile
    assert published.sha256 == sha256(plan_path.read_bytes()).hexdigest()
    assert load_teacher_collection_plan(plan_dir) == published
