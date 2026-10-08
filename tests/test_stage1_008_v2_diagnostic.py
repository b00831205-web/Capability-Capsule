"""Verify prompt-only changes, separate identities and immutable baseline evidence."""

import json
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points
from capability_capsule.eval.records import CaseResult

ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = "stage1-codex-qwen35-2b-008-edit005-64step"
RUN = ROOT / "runs/evaluation" / (MODEL_RUN + "-v2-diagnostic")
BASELINE = ROOT / "runs/evaluation" / MODEL_RUN
GROUPS = {"fixed-v2": "fixed-v3", "heldout-v2": "heldout-v1"}


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_v2_diagnostic_changes_prompt_and_suite_identity_only():
    contract = read(ROOT / "artifacts/sft/stage1-codex-powershell-edit-005-qwen35-2b-v1/manifest.json")["chat_contract"]
    prepared = read(RUN / "prepared.json")
    for group, old_group in GROUPS.items():
        path = RUN / group / "evaluation-suite.json"
        suite = load_coding_evaluation_suite(path)
        old = load_coding_evaluation_suite(BASELINE / old_group / "evaluation-suite.json")
        assert suite.system_prompt == contract["system_prompt"]
        assert suite.system_prompt != old.system_prompt
        assert suite.evaluation_suite_id != old.evaluation_suite_id
        assert suite.cases == old.cases
        assert suite.identity().evaluation_suite_digest == old.identity().evaluation_suite_digest
        assert sha256(path.read_bytes()).hexdigest() == prepared["groups"][group]["suite_sha256"]
        assert prepared["system_prompt_sha256"] == sha256(suite.system_prompt.encode()).hexdigest()


def test_v2_diagnostic_preserves_baseline_training_data_and_fixtures():
    before = read(RUN / "protected-input-hashes.json")
    evidence = read(RUN / "input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert evidence["protected_file_count"] == len(before)
    assert f"runs/evaluation/{MODEL_RUN}/evaluation-summary.json" in before
    assert f"runs/training/{MODEL_RUN}/checkpoint.json" in before
    for relative, digest in before.items():
        assert sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_v2_diagnostic_matches_budgets_and_records_independent_ledgers():
    for group, old_group in GROUPS.items():
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        manifest = read(directory / "run-manifest.json")
        old_manifest = read(BASELINE / old_group / "run-manifest.json")
        for key in ("checkpoint_sha256", "executor_sha256", "tools_sha256", "max_new_tokens", "max_tool_rounds", "do_sample", "case_digest"):
            assert manifest[key] == old_manifest[key]
        assert manifest["system_prompt_sha256"] != old_manifest["system_prompt_sha256"]
        assert manifest["baseline_success_count"] == old_manifest["success_count"]
        assert len(results) == len(suite.cases)
        assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
        assert manifest["success_count"] == sum(r.success for r in results)
        assert manifest["time_bounded_success_count"] == sum(r.time_bounded_success for r in results)
        assert all(r.model_id == f"capsule-{MODEL_RUN}" and r.tool_call_count <= 4 for r in results)
        points = load_learning_curve_points(directory / "learning-curve.jsonl")
        assert len(points) == 1
        assert points[0].training_run_id == MODEL_RUN
        assert points[0].checkpoint_id == MODEL_RUN + "-step-64"
        assert points[0].evaluation_suite_id == suite.evaluation_suite_id
        assert points[0].cumulative_trajectory_count == 12 and points[0].cumulative_token_count == 8169
        assert points[0].success_rate == sum(r.success for r in results) / len(results)
        for result in results:
            report = read(directory / "case-reports" / f"{result.case_id}.json")
            assert report["result"] == result.model_dump(mode="json")
            if result.success:
                assert result.invalid_tool_call_count == 0
                assert all(v["passed"] for v in report["validators"])
