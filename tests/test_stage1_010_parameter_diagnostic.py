"""Matched task-only control, preflight gates, truthful scores and preservation."""

import pytest
import run_stage1_010_parameter_diagnostic as diagnostic
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import CaseResult


def test_parameter_diagnostic_changes_only_task_description_and_bookkeeping_ids():
    fixed, original = diagnostic.source_case()
    first, second = [load_coding_evaluation_suite(diagnostic.RUN / group / "evaluation-suite.json") for group in diagnostic.GROUPS]
    a, b = first.cases[0], second.cases[0]
    assert a.validators == b.validators
    assert a.fixture_root == b.fixture_root
    assert first.system_prompt == second.system_prompt == fixed.system_prompt
    assert a.task.task == original.task.task
    assert b.task.task == original.task.task + diagnostic.CLARIFICATION
    left, right = a.task.model_dump(mode="json"), b.task.model_dump(mode="json")
    assert {key for key in left if left[key] != right[key]} == {"task_id", "task"}
    assert a.task.fixture_revision == b.task.fixture_revision == original.task.fixture_revision
    snapshot = diagnostic.ROOT / a.fixture_root
    assert inspect_fixture(a.task.fixture_id, snapshot).revision == original.task.fixture_revision
    for name in ("greeting.py", "test_greeting.py"):
        assert (snapshot / name).read_bytes() == (diagnostic.ROOT / original.fixture_root / name).read_bytes()
    assert tuple(diagnostic.TOOLS) == tuple(diagnostic.training.baseline.source.contract().tools)


def test_parameter_diagnostic_preflight_discriminates_single_name_hardcoding():
    preflight = diagnostic.read(diagnostic.RUN / "preflight.json")
    assert not preflight["original"][0]["passed"]
    assert all(v["passed"] for v in preflight["reference"])
    hardcoded = {v["validator_id"]: v["passed"] for v in preflight["hardcoded"]}
    assert hardcoded["pytest:test_greeting.py"] and hardcoded["python-assert:dynamic-0"]
    assert not hardcoded["exact-content:greeting.py"]
    assert all(not hardcoded[f"python-assert:dynamic-{i}"] for i in (1, 2, 3))
    assert hardcoded["unchanged:test_greeting.py"]
    assert len(preflight["reference"]) == 7


def test_parameter_diagnostic_reports_actual_outputs_without_unseen_or_promotion_ledger():
    prepared = diagnostic.read(diagnostic.RUN / "prepared.json")
    summary = diagnostic.read(diagnostic.RUN / "diagnostic-summary.json")
    assert "not fresh unseen" in summary["scope"]
    assert prepared["replications_per_group"] == summary["replications_per_group"] == 1
    assert not summary["learning_curve_ledger_written"]
    assert not list(diagnostic.RUN.rglob("learning-curve.jsonl"))
    for group in diagnostic.GROUPS:
        directory = diagnostic.RUN / group
        report = diagnostic.read(directory / "case-report.json")
        manifest = diagnostic.read(directory / "run-manifest.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        assert len(results) == 1 and report["result"] == results[0].model_dump(mode="json")
        assert summary["groups"][group] == manifest
        passes = {v["validator_id"]: v["passed"] for v in report["validators"]}
        assert manifest["strict_success"] == results[0].success
        assert manifest["visible_ada_test_passed"] == passes["pytest:test_greeting.py"]
        assert manifest["multi_name_behavior_passed"] == all(passes[f"python-assert:dynamic-{i}"] for i in range(4))
        assert manifest["parameter_preserving_exact_content_passed"] == passes["exact-content:greeting.py"]
        assert manifest["test_file_unchanged"] == passes["unchanged:test_greeting.py"]
        assert prepared["prepared_at"] <= results[0].started_at.isoformat() < manifest["completed_at"]
        assert manifest["checkpoint_sha256"] == diagnostic.digest(diagnostic.TRAINING / "checkpoint.json")
        assert manifest["suite_sha256"] == diagnostic.digest(directory / "evaluation-suite.json")
        assert manifest["max_new_tokens"] == 256 and manifest["max_tool_rounds"] == 4
        assert not manifest["do_sample"] and not manifest["enable_thinking"]
        assert results[0].tool_call_count <= 4 and len(report["completions"]) <= 5
        if results[0].success:
            assert all(passes.values()) and results[0].invalid_tool_call_count == 0


def test_parameter_diagnostic_preserves_existing_artifacts_and_copied_tests():
    prepared, before = diagnostic.audit_prepared()
    preservation = diagnostic.read(diagnostic.RUN / "input-preservation.json")
    assert preservation["unchanged"] and preservation["changed_paths"] == []
    assert preservation["protected_file_count"] == len(before)
    assert "runs/evaluation/stage1-codex-qwen35-2b-010-literal006-64step-retry1/evaluation-summary.json" in before
    assert (diagnostic.TRAINING / "adapter-final/adapter_model.safetensors").relative_to(diagnostic.ROOT).as_posix() in before
    for relative, pinned in prepared["snapshot_hashes"].items():
        assert diagnostic.digest(diagnostic.ROOT / relative) == pinned
    for group in diagnostic.GROUPS:
        assert (diagnostic.RUN / group / "workspace/test_greeting.py").read_bytes() == (diagnostic.RUN / "fixture/test_greeting.py").read_bytes()


def test_parameter_diagnostic_refuses_repeated_preparation_or_inference():
    summary = (diagnostic.RUN / "diagnostic-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic.prepare()
    with pytest.raises(FileExistsError):
        diagnostic.evaluate()
    assert (diagnostic.RUN / "diagnostic-summary.json").read_bytes() == summary
