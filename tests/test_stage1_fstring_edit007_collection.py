"""Actual raw-only inline f-string collection and preserved provenance."""

import ast
import json
from hashlib import sha256

import pytest
import collect_stage1_fstring_edit007 as collection
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory
from capability_capsule.eval.teacher_collection_plan import pending_teacher_assignments
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_fstring007_six_pinned_train_assignments_complete_and_raw_unique():
    plan = load_teacher_collection_plan(collection.PLAN_DIR).plan
    raw = load_jsonl(collection.RAW, TeacherTrajectory)
    assert len(plan.assignments) == len(raw) == 6
    assert pending_teacher_assignments(plan, artifact_root=collection.ROOT) == ()
    assert len({t.trajectory_id for t in raw}) == 6
    assert {p.name for p in collection.PLAN_DIR.iterdir()} == {"collection-plan.json", "manifest.json"}
    for assignment, record in zip(plan.assignments, raw, strict=True):
        assert record.split is assignment.task.split is DatasetSplit.TRAIN
        assert record.teacher_model == "gpt-5" and record.teacher_skill_version == "0.2.0"
        assert record.trajectory_id == assignment.trajectory_id and record.task_id == assignment.task.task_id
        source = collection.ROOT / assignment.authorized_fixture_root
        assert inspect_fixture(assignment.task.fixture_id, source).revision == record.source_revision == assignment.task.fixture_revision
        assert assignment.student_target == plan.student_target and assignment.harness_profile == plan.harness_profile
    report = collection.audit(plan)
    assert report["validation_count"] == report["known_exact_duplicates"] == 0
    assert not report["validation_exact_overlap"]
    assert report["hardware_verification_status"] == "confirmed"
    assert report["recommendation_evidence_status"] == "provisional"


def test_fstring007_raw_contains_real_bounded_calls_and_no_invented_recovery():
    raw = load_jsonl(collection.RAW, TeacherTrajectory)
    call_count = 0
    for variant, record in zip(collection.VARIANTS, raw, strict=True):
        calls = [call for message in record.messages for call in message.tool_calls]
        results = [json.loads(message.content) for message in record.messages if message.role.value == "tool"]
        assert len(calls) == len(results) in (3, 4)
        assert all(call.name == "exec_command" and set(call.arguments) == {"cmd"} for call in calls)
        assert all(result["exit_code"] == 0 for result in results)
        assert results[0]["output"] == variant.source
        assert "passed" in results[-1]["output"]
        assert calls[-1].arguments["cmd"] == "pytest -q"
        expected_parts = collection.EDIT_PARTS.get(variant.slug, ((variant.old, variant.new),))
        assert len(calls) == len(expected_parts) + 2
        for call, (old, new), result in zip(calls[1:-1], expected_parts, results[1:-1], strict=True):
            assert call.arguments["cmd"] == (
                "$text = Get-Content -LiteralPath greeting.py -Raw; "
                f"$updated = $text.Replace('{old}', '{new}'); Set-Content -LiteralPath greeting.py -Value $updated")
            assert result["output"] == ""
        call_count += len(calls)
    assert call_count == read(collection.EVIDENCE / "collection-audit.json")["observed_tool_calls"] == 21
    notes = read(collection.EVIDENCE / "collection-interruption-notes.json")
    assert notes["incidents"][0]["raw_records_at_interruption"] == 0
    assert notes["incidents"][1]["raw_records_at_interruption"] == 1
    assert not notes["incidents"][1]["source_edit_occurred"]


def test_fstring007_preserves_dynamic_expressions_and_multi_input_tests():
    for variant in collection.VARIANTS:
        source = collection.SOURCES / variant.slug
        workspace = collection.WORKSPACES / variant.slug
        before = (source / "greeting.py").read_text("utf-8")
        after = (workspace / "greeting.py").read_text("utf-8")
        assert before == variant.source and after == before.replace(variant.old, variant.new)
        assert collection.normalized_tree(before) == collection.normalized_tree(after)
        expressions = lambda code: [ast.dump(n) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FormattedValue)]
        assert expressions(before) == expressions(after) and expressions(before)
        assert (source / "test_greeting.py").read_bytes() == (workspace / "test_greeting.py").read_bytes()
        assert {name for name, _ in collection.CHECKS[variant.slug]} >= {variant.name, ""}
        assert len(collection.CHECKS[variant.slug]) >= 3
        assert read(collection.EVIDENCE / "baseline-checks" / f"{variant.slug}.json")["exit_code"] == 1
        # The first trajectory used the prior recorder, which ran an independent validator too.
        evidence = collection.EVIDENCE / "independent-checks" / f"{variant.slug}.json"
        if evidence.exists():
            assert read(evidence)["exit_code"] == 0
            assert read(evidence)["test_sha256"] == sha256((source / "test_greeting.py").read_bytes()).hexdigest()


def test_fstring007_preserves_existing_artifacts_and_stops_at_raw_review():
    before = read(collection.EVIDENCE / "protected-input-hashes.json")
    preservation = read(collection.EVIDENCE / "input-preservation.json")
    assert preservation == {"unchanged": True, "file_count": len(before)}
    current = collection.protected_hashes()
    assert all(current.get(path) == pinned for path, pinned in before.items())
    assert "runs/evaluation/stage1-codex-qwen35-2b-010-literal006-64step-retry1-parameter-diagnostic-v1/diagnostic-summary.json" in before
    report = read(collection.EVIDENCE / "collection-audit.json")
    assert report["independent_validators_passed"] == report["baseline_target_tests_failed"] == 6
    assert report["raw_sha256"] == sha256(collection.RAW.read_bytes()).hexdigest()
    assert not report["published"] and not report["sft_exported"] and not report["training_started"]


def test_fstring007_normalization_allows_literal_suffix_but_rejects_parameter_loss():
    before = 'def greeting(name):\n    return f"Hey {name}"\n'
    changed = 'def greeting(name):\n    return f"Go {name}!"\n'
    hardcoded = 'def greeting(name):\n    return f"Go Wren!"\n'
    assert collection.normalized_tree(before) == collection.normalized_tree(changed)
    assert collection.normalized_tree(before) != collection.normalized_tree(hardcoded)
    assert collection.normalized_tree(before) != collection.normalized_tree(changed.replace("{name}", "{name.upper()}"))


def test_fstring007_refuses_to_overwrite_plan_or_raw(monkeypatch):
    original = collection.RAW.read_bytes()
    monkeypatch.setattr(collection.sys, "argv", ["collect_stage1_fstring_edit007.py"])
    with pytest.raises(FileExistsError):
        collection.main()
    assert collection.RAW.read_bytes() == original
