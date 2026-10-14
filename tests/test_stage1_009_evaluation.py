"""Frozen v2/unseen definitions, genuine first-use provenance and isolated scores."""

import json
from hashlib import sha256

import run_stage1_009_evaluation as evaluation
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points
from capability_capsule.eval.records import CaseResult, DatasetSplit


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_009_evaluation_preserves_exact_fixed_v2_and_generation_contract():
    prepared = read(evaluation.RUN / "prepared.json")
    fixed_file = evaluation.RUN / "fixed-v2/evaluation-suite.json"
    assert fixed_file.read_bytes() == evaluation.SUITE.read_bytes()
    fixed = load_coding_evaluation_suite(fixed_file)
    contract = evaluation.training.source.contract()
    assert fixed.system_prompt == contract.system_prompt
    assert tuple(contract.tools) == evaluation.TOOLS
    assert prepared["checkpoint_sha256"] == evaluation.digest(evaluation.TRAINING / "checkpoint.json")
    assert prepared["executor_sha256"] == evaluation.digest(evaluation.EXECUTOR)
    assert prepared["system_prompt_sha256"] == sha256(fixed.system_prompt.encode()).hexdigest()
    for group, count in (("fixed-v2", 3), ("unseen-v1", 2)):
        directory = evaluation.RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        manifest = read(directory / "run-manifest.json")
        assert len(suite.cases) == manifest["case_count"] == count
        assert suite.system_prompt == fixed.system_prompt
        assert manifest["max_new_tokens"] == 256 and manifest["max_tool_rounds"] == 4
        assert not manifest["do_sample"] and not manifest["enable_thinking"]
        assert manifest["prepared_at"] == prepared["prepared_at"]
        assert manifest["prepared_at"] < manifest["completed_at"]
        assert manifest["suite_sha256"] == evaluation.digest(directory / "evaluation-suite.json")
        assert manifest["case_digest"] == suite.identity().evaluation_suite_digest


def test_009_unseen_is_new_frozen_validation_not_training_or_prior_evaluation():
    directory = evaluation.RUN / "unseen-v1"
    suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
    audit = read(evaluation.RUN / "unseen-overlap-audit.json")
    # Verify the frozen first-use audit, not a retrospective audit against future authorized runs.
    assert audit["cases"] == [{"case_id": case.case_id, "source_revision": case.task.fixture_revision,
        "task_id": case.task.task_id, "source_sha256": evaluation.digest(evaluation.ROOT / case.fixture_root / "greeting.py")}
        for case in suite.cases]
    assert not audit["exact_overlap"] and len(audit["cases"]) == 2
    assert audit["published_training_task_count"] >= 8
    prepared = read(evaluation.RUN / "prepared.json")
    for case, variant in zip(suite.cases, evaluation.UNSEEN, strict=True):
        assert case.task.split is DatasetSplit.VALIDATION
        snapshot = evaluation.ROOT / case.fixture_root
        assert inspect_fixture(case.task.fixture_id, snapshot).revision == case.task.fixture_revision
        assert (snapshot / "greeting.py").read_text("utf-8") == variant[1]
        preflight = read(directory / f"preflight-{variant[0]}.json")
        assert not preflight["original"][0]["passed"]
        assert preflight["execution"]["authorized"] and preflight["execution"]["exit_code"] == 0
        assert all(v["passed"] for v in preflight["corrected"])
        assert [v.kind for v in case.validators] == ["pytest", "exact_text", "exact_text"]
        assert case.validators[1].expected == variant[1].replace(variant[2], variant[3])
        assert case.validators[2].expected == (snapshot / "test_greeting.py").read_text("utf-8")
    for relative, digest in prepared["unseen_snapshot_hashes"].items():
        assert evaluation.digest(evaluation.ROOT / relative) == digest


def test_009_scores_actual_reports_and_keeps_groups_and_ledgers_separate():
    prepared = read(evaluation.RUN / "prepared.json")
    for group in evaluation.GROUPS:
        directory = evaluation.RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        manifest = read(directory / "run-manifest.json")
        assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
        assert manifest["success_count"] == sum(r.success for r in results)
        assert manifest["time_bounded_success_count"] == sum(r.time_bounded_success for r in results)
        behavior_passes = 0
        for result in results:
            report = read(directory / "case-reports" / f"{result.case_id}.json")
            assert report["result"] == result.model_dump(mode="json")
            assert result.started_at.isoformat() >= prepared["prepared_at"]
            assert result.tool_call_count <= 4 and len(report["completions"]) <= 5
            behavior_passes += all(v["passed"] for v in report["validators"] if v["validator_id"].startswith(("pytest:", "python-assert:")))
            if result.success:
                assert result.invalid_tool_call_count == 0
                assert all(v["passed"] for v in report["validators"])
        assert manifest["behavior_pass_count"] == behavior_passes
        points = load_learning_curve_points(directory / "learning-curve.jsonl")
        assert len(points) == 1
        assert points[0].training_run_id == evaluation.RUN_ID
        assert points[0].checkpoint_id == f"{evaluation.RUN_ID}-step-32"
        assert points[0].cumulative_trajectory_count == 8
        assert points[0].cumulative_token_count == 5506
        assert points[0].success_rate == sum(r.success for r in results) / len(results)
        assert points[0].evaluation_suite_id == suite.evaluation_suite_id


def test_009_evaluation_preserves_data_training_old_runs_and_new_snapshots():
    before = read(evaluation.RUN / "protected-input-hashes.json")
    evidence = read(evaluation.RUN / "input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert evidence["protected_file_count"] == len(before)
    current = evaluation.protected_hashes()
    assert all(current.get(path) == digest for path, digest in before.items())
    assert (evaluation.TRAINING / "checkpoint.json").relative_to(evaluation.ROOT).as_posix() in before
    assert (evaluation.TRAINING / "adapter-final/adapter_model.safetensors").relative_to(evaluation.ROOT).as_posix() in before
    assert "runs/evaluation/stage1-codex-qwen35-2b-008-edit005-64step-v2-diagnostic/evaluation-summary.json" in before


def test_009_failure_is_envelope_copy_and_command_policy_not_read_or_structure_only():
    for group in evaluation.GROUPS:
        directory = evaluation.RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        assert all(not r.success and r.error_type == "ToolRoundLimitExceeded" for r in results)
        assert all(r.tool_call_count == 4 and r.invalid_tool_call_count == 3 for r in results)
        assert read(directory / "run-manifest.json")["behavior_pass_count"] == 0
        for case in suite.cases:
            report = read(directory / "case-reports" / f"{case.case_id}.json")
            assert report["tool_results"][0]["authorized"] and report["tool_results"][0]["exit_code"] == 0
            assert all(not tool["authorized"] and tool["exit_code"] == 126 for tool in report["tool_results"][1:])
            assert '"envelope"' in report["completions"][1]
            assert not report["validators"][0]["passed"]
            assert inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision == case.task.fixture_revision
