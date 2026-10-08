"""Integrity checks for the independently scored fixed and heldout groups."""

import json
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite, parse_qwen_tool_completion
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points
from capability_capsule.eval.records import CaseResult

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-007-edit005-32step"


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_007_suites_are_pinned_before_inference_and_heldout_is_validation_only():
    fixed = load_coding_evaluation_suite(RUN / "fixed-v3/evaluation-suite.json")
    baseline = load_coding_evaluation_suite(
        ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy-raw-order/evaluation-suite.json")
    heldout = load_coding_evaluation_suite(RUN / "heldout-v1/evaluation-suite.json")
    prepared = read(RUN / "prepared.json")
    assert fixed == baseline
    assert len(fixed.cases) == 3 and len(heldout.cases) == 2
    assert fixed.system_prompt == heldout.system_prompt
    assert fixed.identity().evaluation_suite_digest != heldout.identity().evaluation_suite_digest
    assert fixed.evaluation_suite_id != heldout.evaluation_suite_id
    audit = read(RUN / "heldout-overlap-audit.json")
    assert audit["published_training_revision_count"] >= 12
    assert audit["published_training_task_count"] >= 12
    assert audit["exact_revision_task_source_overlap"] is False
    for group in ("fixed-v3", "heldout-v1"):
        path = RUN / group / "evaluation-suite.json"
        assert sha256(path.read_bytes()).hexdigest() == prepared["groups"][group]["suite_sha256"]
    for case in heldout.cases:
        assert case.task.split.value == "validation"
        snapshot = ROOT / case.fixture_root
        assert inspect_fixture(case.task.fixture_id, snapshot).revision == case.task.fixture_revision
        assert [v.kind for v in case.validators] == ["pytest", "exact_text", "exact_text"]
        assert case.validators[-1].expected == (snapshot / "test_greeting.py").read_text("utf-8")
    for slug in ("mapping-template", "tuple-join"):
        preflight = read(RUN / "heldout-v1" / f"preflight-{slug}.json")
        assert not preflight["original"][0]["passed"]
        assert preflight["execution"]["authorized"]
        assert preflight["execution"]["exit_code"] == 0
        assert all(v["passed"] for v in preflight["corrected"])


def test_007_evaluation_preserves_inputs_and_scores_separate_ledgers():
    preservation = read(RUN / "input-preservation.json")
    assert preservation["unchanged"] is True
    assert preservation["changed_paths"] == []
    before = read(RUN / "protected-input-hashes.json")
    assert preservation["protected_file_count"] == len(before)
    for relative, digest in before.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for group, count in (("fixed-v3", 3), ("heldout-v1", 2)):
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        points = load_learning_curve_points(directory / "learning-curve.jsonl")
        manifest = read(directory / "run-manifest.json")
        assert len(results) == count
        assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
        assert manifest["success_count"] == sum(r.success for r in results)
        assert manifest["case_count"] == count
        assert manifest["max_new_tokens"] == 256
        assert manifest["max_tool_rounds"] == 4
        assert manifest["do_sample"] is False
        assert manifest["prepared_at"] < manifest["completed_at"]
        assert manifest["case_digest"] == suite.identity().evaluation_suite_digest
        assert manifest["system_prompt_sha256"] == sha256(suite.system_prompt.encode()).hexdigest()
        assert len(points) == 1
        assert points[0].training_run_id == RUN.name
        assert points[0].cumulative_trajectory_count == 12
        assert points[0].cumulative_token_count == 8169
        assert points[0].success_rate == sum(r.success for r in results) / count
        assert points[0].evaluation_suite_id == suite.evaluation_suite_id
        assert points[0].evaluation_suite_digest == manifest["case_digest"]
        for result in results:
            report = read(directory / "case-reports" / f"{result.case_id}.json")
            assert report["result"] == result.model_dump(mode="json")
            assert result.tool_call_count <= 4
            if result.success:
                assert result.invalid_tool_call_count == 0
                assert all(v["passed"] for v in report["validators"])


def test_007_failure_is_rejected_edit_not_read_or_fixture_corruption():
    for group, invalid in (("fixed-v3", [3, 2, 3]), ("heldout-v1", [3, 3])):
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        assert [r.invalid_tool_call_count for r in results] == invalid
        assert all(not r.success and r.error_type == "ToolRoundLimitExceeded" for r in results)
        for case in suite.cases:
            report = read(directory / "case-reports" / f"{case.case_id}.json")
            assert report["tool_results"][0]["authorized"]
            assert report["tool_results"][0]["exit_code"] == 0
            assert inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision == case.task.fixture_revision
            commands = [call.arguments["cmd"] for text in report["completions"]
                        for call in parse_qwen_tool_completion(text).tool_calls]
            assert any("-Append" in cmd for cmd in commands)
            assert not any(".Replace(" in cmd for cmd in commands)
            assert not report["validators"][0]["passed"]
