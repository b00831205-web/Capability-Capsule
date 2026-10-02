import json
from pathlib import Path

from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.curation import trajectory_fingerprint
from capability_capsule.eval.dataset_pipeline import curate_teacher_dataset
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory
from capability_capsule.eval.teacher_collection_plan import pending_teacher_assignments
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "stage1-codex-powershell-edit-005"


def test_edit005_plan_and_raw_are_complete_and_train_only() -> None:
    published = load_teacher_collection_plan(ROOT / "plans/teacher" / PLAN_ID)
    plan = published.plan
    raw = load_jsonl(ROOT / "datasets/teacher" / PLAN_ID / "raw.jsonl", TeacherTrajectory)
    harness = verify_harness_profile(plan.harness_profile, artifact_root=ROOT)

    assert len(plan.assignments) == len(raw) == 12
    assert pending_teacher_assignments(plan, artifact_root=ROOT) == ()
    assert all(assignment.task.split is DatasetSplit.TRAIN for assignment in plan.assignments)
    assert all(trajectory.split is DatasetSplit.TRAIN for trajectory in raw)
    assert len({item.trajectory_id for item in raw}) == 12
    validate_teacher_dataset(raw, tasks=[item.task for item in plan.assignments], harness_profile=harness)

    for assignment, trajectory in zip(plan.assignments, raw, strict=True):
        source = ROOT / assignment.authorized_fixture_root
        snapshot = inspect_fixture(assignment.task.fixture_id, source)
        assert snapshot.revision == assignment.task.fixture_revision
        assert trajectory.source_revision == snapshot.revision
        assert trajectory.task_id == assignment.task.task_id
        calls = [call for message in trajectory.messages for call in message.tool_calls]
        results = [message for message in trajectory.messages if message.role.value == "tool"]
        assert len(calls) == len(results) == 3
        assert all(call.name == "exec_command" for call in calls)
        assert all(json.loads(message.content)["exit_code"] == 0 for message in results)
        assert ".Replace(" in calls[1].arguments["cmd"]
        assert "Set-Content" in calls[1].arguments["cmd"]
        assert calls[2].arguments["cmd"] == "pytest -q"


def test_edit005_preserves_originals_and_excludes_validation_answers() -> None:
    base = ROOT / "artifacts/teacher-fixtures/train-greeting-good-day"
    assert inspect_fixture("train-greeting-good-day-contract-v1", base).revision == (
        "fc257a990228187dc3e707a0e9acfe8799487eb5b1166b17d75007d89dcb91ce"
    )
    raw_path = ROOT / "datasets/teacher" / PLAN_ID / "raw.jsonl"
    raw_text = raw_path.read_text("utf-8")
    assert "Welcome, Ada" not in raw_text
    assert "PowerShell ready" not in raw_text
    assert "validation-powershell-salutation" not in raw_text

    published = load_teacher_collection_plan(ROOT / "plans/teacher" / PLAN_ID)
    for assignment in published.plan.assignments:
        slug = Path(assignment.authorized_fixture_root).name
        workspace = ROOT / "tmp/teacher" / PLAN_ID / "workspaces" / slug
        source = ROOT / assignment.authorized_fixture_root
        assert (workspace / "test_greeting.py").read_bytes() == (
            source / "test_greeting.py"
        ).read_bytes()
        assert (workspace / "greeting.py").read_bytes() != (
            source / "greeting.py"
        ).read_bytes()


def test_edit005_publication_gate_has_no_known_exact_duplicate_or_validation_leak() -> None:
    plan = load_teacher_collection_plan(ROOT / "plans/teacher" / PLAN_ID).plan
    new = load_jsonl(ROOT / "datasets/teacher" / PLAN_ID / "raw.jsonl", TeacherTrajectory)
    published_train = []
    for path in (ROOT / "datasets/teacher/published").glob("*/train.jsonl"):
        if path.parent.name == PLAN_ID:
            continue
        published_train.extend(load_jsonl(path, TeacherTrajectory))
    old_fingerprints = {trajectory_fingerprint(item) for item in published_train}
    new_fingerprints = [trajectory_fingerprint(item) for item in new]

    assert len(new_fingerprints) == len(set(new_fingerprints)) == 12
    assert not old_fingerprints.intersection(new_fingerprints)
    curated = curate_teacher_dataset(new, tasks=[item.task for item in plan.assignments])
    assert len(curated.train) == 12
    assert curated.validation == ()
    assert curated.duplicates == ()
    assert not {item.trajectory_id for item in new}.intersection(
        item.trajectory_id for item in published_train
    )

    validation = load_coding_evaluation_suite(
        ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
    )
    validation_tasks = {case.task.task.casefold() for case in validation.cases}
    validation_revisions = {case.task.fixture_revision for case in validation.cases}
    validation_groups = {case.task.split_group for case in validation.cases}
    assert all(assignment.task.task.casefold() not in validation_tasks for assignment in plan.assignments)
    assert all(assignment.task.fixture_revision not in validation_revisions for assignment in plan.assignments)
    assert all(assignment.task.split_group not in validation_groups for assignment in plan.assignments)
    assert len({(ROOT / assignment.authorized_fixture_root / "greeting.py").read_bytes()
                for assignment in plan.assignments}) == 12

    for assignment, trajectory in zip(plan.assignments, new, strict=True):
        source = ROOT / assignment.authorized_fixture_root
        tool_results = [message for message in trajectory.messages if message.role.value == "tool"]
        assert json.loads(tool_results[0].content)["output"] == (
            source / "greeting.py"
        ).read_text("utf-8")
        assert json.loads(tool_results[1].content)["output"] == ""
        assert "passed" in json.loads(tool_results[2].content)["output"]


def test_edit005_publication_matches_reviewed_raw_and_manifest() -> None:
    verified = load_teacher_dataset_publication(ROOT / "datasets/teacher/published" / PLAN_ID)
    raw = load_jsonl(ROOT / "datasets/teacher" / PLAN_ID / "raw.jsonl", TeacherTrajectory)
    assert verified.dataset.train == raw
    assert verified.dataset.validation == ()
    assert verified.dataset.duplicates == ()
    assert [artifact.record_count for artifact in verified.manifest.artifacts] == [12, 0, 0]
    assert verified.manifest.artifacts[0].sha256 == (
        "86e537f9ec6d1f0459c98f6efa82f92f0a7ea43fc4e13423d702e6afb2abcda2"
    )
