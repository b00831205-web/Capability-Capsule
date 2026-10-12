"""Read-only acceptance checks for the separately authorized evalfix diagnostic."""

import run_stage1_mix012_fixed_train_fit_diagnostic as probe
from authorized_planning_history import input_matches

read, digest = probe.diagnostic.read, probe.diagnostic.digest


def test_plan_preserves_seen_tasks_and_all_inference_budgets():
    plan = read(probe.RUN / "diagnostic-plan.json")
    prior = read(probe.PRIOR / "diagnostic-plan.json")
    assert all(plan[key] == prior[key] for key in probe.candidate.SAME_FIELDS)
    assert plan["checkpoint_sha256"] == probe.candidate.CHECKPOINT_SHA256
    assert digest(probe.candidate.CHECKPOINT) == plan["checkpoint_sha256"]
    assert digest(probe.FIXED_PATH) == plan["implementation_sha256"] == probe.FIXED_SHA256
    assert digest(probe.ROOT / "tests/run_stage1_mix012_fixed_train_fit_diagnostic.py") == plan["runner_sha256"]
    assert digest(probe.PRIOR / "diagnostic-plan.json") == plan["comparison_plan_sha256"]
    assert digest(probe.PRIOR / "summary.json") == plan["comparison_summary_sha256"]
    assert plan["case_count"] == 14 and all(c["split"] == "train" for c in plan["cases"])


def test_summary_is_derived_from_reports_not_historical_rescoring():
    plan = read(probe.RUN / "diagnostic-plan.json")
    reports = [read(probe.RUN / "case-reports" / (c["task_id"] + ".json")) for c in plan["cases"]]
    summary = read(probe.RUN / "summary.json")
    assert len(reports) == 14
    assert all(summary[k] == v for k,v in probe.diagnostic.summarize(reports).items())
    for report in reports:
        assert report["split"] == "train" and report["seen_during_training"]
        assert report["tool_call_count"] <= 4 and report["generation_turn_count"] <= 5
        assert report["generation_turn_count"] == len(report["turns"])
        assert report["truncated_turn_count"] == sum(t["finish_reason"] == "length" for t in report["turns"])
        assert report["format_error_count"] == sum(t["format_error"] is not None for t in report["turns"])
        assert report["final_edit_incorrect"] == (not report["exact_source_pass"])
        assert report["strict_success"] == (report["reached_final"] and report["error"] is None
            and report["tests_unchanged"] and report["exact_source_pass"] and report["behavior_pass"]
            and report["denied_command_count"] == 0)
        for turn in report["turns"]:
            assert turn["output_tokens"] <= 256
            if turn["finish_reason"] != "eos" or turn["format_error"] is not None:
                assert "execution" not in turn
        if report["termination_reason"] == "generation_length":
            assert not report["reached_final"] and report["truncated_turn_count"] > 0
    assert not summary["checkpoint_promoted"]
    comparison = read(probe.RUN / "comparison.json")
    assert comparison["baseline"] == read(probe.PRIOR / "summary.json")
    assert comparison["candidate"] == summary and len(comparison["pairs"]) == 14


def test_old_data_runs_and_source_bytes_are_preserved():
    protected = read(probe.RUN / "protected-input-hashes.json")
    assert len(protected) > 1378
    assert all(digest(probe.ROOT / p) == value for p,value in protected.items())
    preservation = read(probe.RUN / "input-preservation.json")
    assert preservation["unchanged"] and preservation["changed_paths"] == []
    assert preservation["protected_file_count"] == len(protected)


def test_resource_and_completion_receipts_are_consistent():
    end = read(probe.RUN / "supervisor-result.json")
    assert end["exit_code"] == 0 and end["resource_abort"] is None
    assert not end["model_restarted"]
    auth = read(probe.RUN / "fixed-diagnostic-authorization.json")
    assert not auth["memory_guard_override"] and auth["disk_guard_retained"]
    samples = probe.candidate.preparation.records(probe.RUN / "resource-monitor.jsonl")
    assert samples and all(not probe.safety.below_limits(s, probe.safety.STOP_LIMITS) for s in samples)
    completion = read(probe.RUN / "completion.json")
    assert completion["case_count"] == 14 and not completion["new_training_started"]
    assert not completion["checkpoint_promoted"]
    update = read(probe.RUN / "planning-update.json")
    assert update["explicitly_authorized"]
    for name,before in update["before"].items():
        current = digest(probe.ROOT / name)
        assert update["after"][name] == current
        assert input_matches(name, before, current)
