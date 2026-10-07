"""Frozen first-use provenance, fixed comparison, scores and immutable inputs."""

import json
from hashlib import sha256

import pytest
import run_stage1_010_evaluation as evaluation
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points
from capability_capsule.eval.records import CaseResult, DatasetSplit


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_010_uses_identical_fixed_v2_and_generation_contract():
    prepared = read(evaluation.RUN / "prepared.json")
    fixed_path = evaluation.RUN / "fixed-v2/evaluation-suite.json"
    baseline_path = evaluation.ROOT / "runs/evaluation/stage1-codex-qwen35-2b-009-literal006-32step/fixed-v2/evaluation-suite.json"
    assert fixed_path.read_bytes() == evaluation.SUITE.read_bytes() == baseline_path.read_bytes()
    fixed = load_coding_evaluation_suite(fixed_path)
    contract = evaluation.training.baseline.source.contract()
    assert fixed.system_prompt == contract.system_prompt and tuple(contract.tools) == evaluation.TOOLS
    assert prepared["checkpoint_sha256"] == evaluation.digest(evaluation.TRAINING / "checkpoint.json")
    assert prepared["executor_sha256"] == evaluation.digest(evaluation.EXECUTOR)
    assert prepared["system_prompt_sha256"] == sha256(fixed.system_prompt.encode()).hexdigest()
    assert prepared["chat_contract_sha256"] == contract.sha256()
    for group, count in (("fixed-v2", 3), ("unseen-v1", 2)):
        directory = evaluation.RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        manifest = read(directory / "run-manifest.json")
        assert len(suite.cases) == manifest["case_count"] == count
        assert suite.system_prompt == fixed.system_prompt
        assert manifest["max_new_tokens"] == 256 and manifest["max_tool_rounds"] == 4
        assert not manifest["do_sample"] and not manifest["enable_thinking"]
        assert manifest["prepared_at"] == prepared["prepared_at"] < manifest["completed_at"]
        assert manifest["suite_sha256"] == evaluation.digest(directory / "evaluation-suite.json")
        assert manifest["case_digest"] == suite.identity().evaluation_suite_digest


def test_010_unseen_snapshots_are_frozen_first_use_validation():
    directory = evaluation.RUN / "unseen-v1"
    suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
    audit = read(evaluation.RUN / "unseen-overlap-audit.json")
    assert audit["cases"] == [{"case_id": case.case_id, "source_revision": case.task.fixture_revision,
        "task_id": case.task.task_id, "source_sha256": evaluation.digest(evaluation.ROOT / case.fixture_root / "greeting.py")}
        for case in suite.cases]
    assert not audit["exact_overlap"] and len(audit["cases"]) == 2
    assert audit["previous_validation_case_count"] >= 7
    prior = load_coding_evaluation_suite(evaluation.ROOT / "runs/evaluation/stage1-codex-qwen35-2b-009-literal006-32step/unseen-v1/evaluation-suite.json")
    assert not {c.case_id for c in prior.cases} & {c.case_id for c in suite.cases}
    assert not {c.task.fixture_revision for c in prior.cases} & {c.task.fixture_revision for c in suite.cases}
    for case, variant in zip(suite.cases, evaluation.UNSEEN, strict=True):
        snapshot = evaluation.ROOT / case.fixture_root
        assert case.task.split is DatasetSplit.VALIDATION
        assert inspect_fixture(case.task.fixture_id, snapshot).revision == case.task.fixture_revision
        assert (snapshot / "greeting.py").read_text("utf-8") == variant[1]
        preflight = read(directory / f"preflight-{variant[0]}.json")
        assert not preflight["original"][0]["passed"]
        assert preflight["execution"]["authorized"] and preflight["execution"]["exit_code"] == 0
        assert all(v["passed"] for v in preflight["corrected"])
        assert [v.kind for v in case.validators] == ["pytest", "exact_text", "exact_text"]
        assert case.validators[1].expected == variant[1].replace(variant[2], variant[3])
        assert case.validators[2].expected == (snapshot / "test_greeting.py").read_text("utf-8")
    for relative, pinned in read(evaluation.RUN / "prepared.json")["unseen_snapshot_hashes"].items():
        assert evaluation.digest(evaluation.ROOT / relative) == pinned


def test_010_reports_actual_scores_and_separate_64step_ledgers():
    prepared = read(evaluation.RUN / "prepared.json")
    summary = read(evaluation.RUN / "evaluation-summary.json")
    for group in evaluation.GROUPS:
        directory = evaluation.RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        manifest = read(directory / "run-manifest.json")
        assert summary[group] == manifest
        assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
        assert manifest["success_count"] == sum(r.success for r in results)
        assert manifest["time_bounded_success_count"] == sum(r.time_bounded_success for r in results)
        behavior = 0
        for result in results:
            report = read(directory / "case-reports" / f"{result.case_id}.json")
            assert report["result"] == result.model_dump(mode="json")
            assert result.started_at.isoformat() >= prepared["prepared_at"]
            assert result.tool_call_count <= 4 and len(report["completions"]) <= 5
            behavior += all(v["passed"] for v in report["validators"] if v["validator_id"].startswith(("pytest:", "python-assert:")))
            if result.success:
                assert result.invalid_tool_call_count == 0 and all(v["passed"] for v in report["validators"])
        assert manifest["behavior_pass_count"] == behavior
        points = load_learning_curve_points(directory / "learning-curve.jsonl")
        assert len(points) == 1
        assert points[0].training_run_id == evaluation.RUN_ID
        assert points[0].checkpoint_id == f"{evaluation.RUN_ID}-step-64"
        assert points[0].cumulative_trajectory_count == 8 and points[0].cumulative_token_count == 5506
        assert points[0].success_rate == sum(r.success for r in results) / len(results)
        assert points[0].evaluation_suite_digest == suite.identity().evaluation_suite_digest
    assert summary["fixed-v2"]["success_count"] == 2
    assert summary["fixed-v2"]["behavior_pass_count"] == 3
    assert summary["unseen-v1"]["success_count"] == 2
    failed = read(evaluation.RUN / "fixed-v2/case-reports/validation-powershell-salutation-unseen-001.json")
    assert not failed["result"]["success"] and failed["result"]["invalid_tool_call_count"] == 0
    assert failed["validators"][0]["passed"] and not failed["validators"][1]["passed"]
    assert "'PowerShell ready, Ada!'" in failed["completions"][1]


def test_010_preserves_old_data_training_evaluation_and_new_snapshots():
    before = read(evaluation.RUN / "protected-input-hashes.json")
    report = read(evaluation.RUN / "input-preservation.json")
    assert report["unchanged"] and report["changed_paths"] == []
    assert report["protected_file_count"] == len(before)
    current = evaluation.protected_hashes()
    assert all(current.get(path) == pinned for path, pinned in before.items())
    for relative in ("checkpoint.json", "adapter-final/adapter_model.safetensors", "optimizer-log.jsonl"):
        assert (evaluation.TRAINING / relative).relative_to(evaluation.ROOT).as_posix() in before
    assert "runs/evaluation/stage1-codex-qwen35-2b-009-literal006-32step/evaluation-summary.json" in before
    assert evaluation.training.verify_preservation()
    prepared = read(evaluation.RUN / "prepared.json")
    for group in evaluation.GROUPS:
        directory = evaluation.RUN / group
        for case in load_coding_evaluation_suite(directory / "evaluation-suite.json").cases:
            relative = prepared["fixed_fixture_sources"][case.case_id] if group == "fixed-v2" else case.fixture_root
            assert (directory / "workspaces" / case.case_id / "test_greeting.py").read_bytes() == (
                evaluation.ROOT / relative / "test_greeting.py").read_bytes()


def test_010_refuses_to_overwrite_evaluation():
    summary = (evaluation.RUN / "evaluation-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.prepare()
    assert (evaluation.RUN / "evaluation-summary.json").read_bytes() == summary
