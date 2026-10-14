"""Raw review only: no published dataset, SFT or model-readiness claims."""

import ast
import json
from hashlib import sha256

import collect_stage1_literal_edit006 as collection
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory
from capability_capsule.eval.teacher_collection_plan import pending_teacher_assignments
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_literal006_complete_pinned_train_assignments_and_raw():
    plan = load_teacher_collection_plan(collection.PLAN_DIR).plan
    report = collection.audit(plan)
    assert report["train_count"] == 8
    assert report["validation_count"] == report["known_exact_duplicates"] == 0
    assert report["hardware_verification_status"] == "confirmed"
    assert report["recommendation_evidence_status"] == "provisional"
    assert pending_teacher_assignments(plan, artifact_root=collection.ROOT) == ()
    trajectories = load_jsonl(collection.RAW, TeacherTrajectory)
    assert len({t.trajectory_id for t in trajectories}) == 8
    assert all(t.split is DatasetSplit.TRAIN for t in trajectories)
    for assignment, trajectory in zip(plan.assignments, trajectories, strict=True):
        assert trajectory.teacher_model == "gpt-5"
        assert trajectory.teacher_skill_version == "0.2.0"
        source = collection.ROOT / assignment.authorized_fixture_root
        assert inspect_fixture(assignment.task.fixture_id, source).revision == trajectory.source_revision
        assert trajectory.source_revision == assignment.task.fixture_revision
        assert set(assignment.task.validation_ids) == {"pytest:test_greeting.py", "exact-local-literal-edit", "unchanged-test-file"}


def test_literal006_observed_commands_are_literal_only_and_results_are_real():
    trajectories = load_jsonl(collection.RAW, TeacherTrajectory)
    for variant, trajectory in zip(collection.VARIANTS, trajectories, strict=True):
        calls = [call for message in trajectory.messages for call in message.tool_calls]
        results = [json.loads(message.content) for message in trajectory.messages if message.role.value == "tool"]
        assert len(calls) == len(results) == 3
        assert all(call.name == "exec_command" and set(call.arguments) == {"cmd"} for call in calls)
        assert all(result["exit_code"] == 0 for result in results)
        assert results[0]["output"] == variant.source
        assert results[1]["output"] == ""
        assert "passed" in results[2]["output"]
        assert calls[1].arguments["cmd"] == (
            "$text = Get-Content -LiteralPath greeting.py -Raw; "
            f"$updated = $text.Replace('{variant.old}', '{variant.new}'); "
            "Set-Content -LiteralPath greeting.py -Value $updated")
        assert calls[2].arguments["cmd"] == "pytest -q"


def test_literal006_preserves_python_structure_tests_and_pinned_snapshots():
    for variant in collection.VARIANTS:
        source = collection.SOURCES / variant.slug
        workspace = collection.WORKSPACES / variant.slug
        before = (source / "greeting.py").read_text("utf-8")
        after = (workspace / "greeting.py").read_text("utf-8")
        assert before == variant.source
        assert before.count(variant.old) == 1
        assert after == before.replace(variant.old, variant.new)
        assert (workspace / "test_greeting.py").read_bytes() == (source / "test_greeting.py").read_bytes()
        assert read(collection.EVIDENCE / "baseline-checks" / f"{variant.slug}.json")["exit_code"] == 1
        # Normalize only literal constants. All lookups, joins, calls and branch structure stay identical.
        trees = [ast.parse(code) for code in (before, after)]
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = "<string>"
        assert ast.dump(trees[0]) == ast.dump(trees[1])


def test_literal006_preserves_existing_artifacts_and_records_collection_scope():
    before = read(collection.EVIDENCE / "protected-input-hashes.json")
    evidence = read(collection.EVIDENCE / "input-preservation.json")
    assert evidence == {"unchanged": True, "file_count": len(before)}
    current = collection.protected_hashes()
    assert all(current.get(path) == digest for path, digest in before.items())
    for relative, digest in before.items():
        assert sha256((collection.ROOT / relative).read_bytes()).hexdigest() == digest
    report = read(collection.EVIDENCE / "collection-audit.json")
    assert report["observed_tool_calls"] == 24
    assert report["independent_validators_passed"] == report["baseline_target_tests_failed"] == 8
    assert not report["published"] and not report["sft_exported"] and not report["training_started"]
    # The immutable collection report describes that historical phase, not later publication.


def test_literal006_has_no_known_exact_overlap_with_validation():
    plan = load_teacher_collection_plan(collection.PLAN_DIR).plan
    suites = [collection.ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json",
              collection.ROOT / "runs/evaluation/stage1-codex-qwen35-2b-008-edit005-64step-v2-diagnostic/heldout-v2/evaluation-suite.json"]
    validation_tasks, validation_revisions, groups = set(), set(), set()
    for path in suites:
        for case in load_coding_evaluation_suite(path).cases:
            validation_tasks.add(case.task.task.casefold())
            validation_revisions.add(case.task.fixture_revision)
            groups.add(case.task.split_group)
    assert all(a.task.task.casefold() not in validation_tasks for a in plan.assignments)
    assert all(a.task.fixture_revision not in validation_revisions for a in plan.assignments)
    assert all(a.task.split_group not in groups for a in plan.assignments)
    raw = collection.RAW.read_text("utf-8")
    assert all(target not in raw for target in ("Welcome, Ada", "PowerShell ready", "Safe travels", "Welcome aboard"))
