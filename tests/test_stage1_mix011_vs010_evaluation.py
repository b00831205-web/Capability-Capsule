"""Preregistered identical cases, actual paired reports and immutable prior artifacts."""

import json

import pytest
import run_stage1_mix011_vs010_evaluation as evaluation
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import CaseResult
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_paired_evaluation_preserves_all_suite_bytes_and_preselected_models():
    prepared = evaluation.verify_prepared()
    assert prepared["first_use_before_model_inference"] and prepared["model_evaluation_authorized"]
    assert not prepared["root_planning_edits_authorized"]
    assert prepared["max_new_tokens"] == 256 and prepared["max_tool_rounds"] == 4
    assert not prepared["do_sample"] and not prepared["enable_thinking"]
    for side, groups in evaluation.GROUPS.items():
        checkpoint = read(evaluation.MODELS[side] / "checkpoint.json")
        assert checkpoint["base_model_revision"] == evaluation.training.frozen.mix.baseline.REVISION
        assert checkpoint["step"] == (28 if side == "011" else 64)
        for group in groups:
            directory = evaluation.RUN / side / group
            source = evaluation.training.frozen.FIXED if group == "fixed-v2" else evaluation.FROZEN / "evaluation-suite.json"
            assert (directory / "evaluation-suite.json").read_bytes() == source.read_bytes()
            suite = load_coding_evaluation_suite(source)
            for case in suite.cases:
                assert case.task.split.value == "validation"
                # Model edits are confined to independent copies, never initial sources.
                origin = evaluation.ROOT / prepared["fixture_sources"][f"{side}/{group}/{case.case_id}"]
                assert inspect_fixture(case.task.fixture_id, origin).revision == case.task.fixture_revision
    assert (evaluation.RUN / "011/frozen-six/evaluation-suite.json").read_bytes() == (
        evaluation.RUN / "010/frozen-six/evaluation-suite.json").read_bytes()
    for relative,pinned in prepared["initial_workspace_hashes"].items():
        if relative.endswith("/test_greeting.py"):
            assert evaluation.digest(evaluation.RUN / relative) == pinned


def test_each_group_reports_real_once_only_results_and_separate_correct_ledgers():
    summary = read(evaluation.RUN / "comparison-summary.json")
    for side, groups in evaluation.GROUPS.items():
        attempt = read(evaluation.RUN / side / "inference-attempt.json")
        assert attempt["one_shot"]
        checkpoint = read(evaluation.MODELS[side] / "checkpoint.json")
        for group in groups:
            directory = evaluation.RUN / side / group
            suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
            results = load_jsonl(directory / "case-results.jsonl", CaseResult)
            reports = [read(directory / "case-reports" / (r.case_id + ".json")) for r in results]
            assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
            assert all(r.repetition == 1 and r.tool_call_count <= 4 for r in results)
            manifest = read(directory / "run-manifest.json")
            assert manifest == summary["summaries"][side][group]
            for key,value in evaluation.report_metrics(results,reports).items():
                assert manifest[key] == value
            for result,report in zip(results,reports,strict=True):
                assert report["result"] == result.model_dump(mode="json")
                assert report["tests_byte_unchanged"] and len(report["completions"]) <= 5
                if result.success:
                    assert result.invalid_tool_call_count == 0 and all(v["passed"] for v in report["validators"])
            points = load_learning_curve_points(directory / "learning-curve.jsonl")
            assert len(points) == 1
            point = points[0]
            assert point.training_run_id == checkpoint["run_id"]
            assert point.cumulative_trajectory_count == (14 if side == "011" else 8)
            assert point.cumulative_token_count == (9979 if side == "011" else 5506)
            assert point.success_rate == sum(r.success for r in results)/len(results)
            assert point.evaluation_suite_digest == suite.identity().evaluation_suite_digest
            if group == "frozen-six":
                assert {g:m["case_count"] for g,m in manifest["subgroups"].items()} == {"fstring":2,"structure":2,"business-transfer":2}


def test_comparison_uses_identical_six_cases_and_no_posthoc_promotion():
    summary = read(evaluation.RUN / "comparison-summary.json")
    assert summary["heldout_independence_consumed"] and not summary["checkpoint_promoted"]
    assert summary["old_010_fixed_score_is_historical_not_rerun"]
    assert len(summary["paired_cases"]) == 6
    assert summary["wins_011"] == sum(r["011"]["success"] and not r["010"]["success"] for r in summary["paired_cases"])
    assert summary["wins_010"] == sum(r["010"]["success"] and not r["011"]["success"] for r in summary["paired_cases"])
    assert summary["strict_success_delta"] == summary["wins_011"]-summary["wins_010"]
    for row in summary["paired_cases"]:
        for side in evaluation.MODELS:
            report = read(evaluation.RUN / side / "frozen-six/case-reports" / (row["case_id"] + ".json"))
            assert row[side]["success"] == report["result"]["success"]


def test_evaluation_preserves_old_files_and_refuses_preparation_or_inference_reruns():
    before = read(evaluation.RUN / "protected-input-hashes.json")
    current = evaluation.protected_hashes()
    assert all(current.get(path) == pinned for path,pinned in before.items())
    evidence = read(evaluation.RUN / "input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert evidence["protected_file_count"] == len(before)
    assert evidence["root_planning_files_unchanged"] and evidence["frozen_suite_unchanged"]
    with pytest.raises(FileExistsError):
        evaluation.prepare()
    for side in evaluation.MODELS:
        with pytest.raises(FileExistsError):
            evaluation.evaluate(side)
