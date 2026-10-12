"""Unchanged 011 protocol, one-shot controls and real 012 diagnostic evidence."""

import pytest
import run_stage1_mix012_train_fit_diagnostic as run


@pytest.fixture(autouse=True)
def restore_shared_globals(monkeypatch):
    for name in ("RUN_ID", "CHECKPOINT", "PROPOSED_OUTPUT", "PINS", "build_plan"):
        monkeypatch.setattr(run.preparation, name, getattr(run.preparation, name))
    monkeypatch.setattr(run.diagnostic, "RUN", run.diagnostic.RUN)
    for name in ("RUN", "__file__", "IGNORE_MEMORY_GUARD"):
        monkeypatch.setattr(run.training, name, getattr(run.training, name))


def test_012_plan_preserves_all_011_cases_prompts_and_inference_budgets():
    plan = run.build_plan()
    old = run.diagnostic.read(run.BASELINE / "diagnostic-plan.json")
    assert all(plan[key] == old[key] for key in run.SAME_FIELDS)
    assert plan["checkpoint_id"] == run.TRAIN_ID and plan["checkpoint_step"] == 56
    assert plan["checkpoint_sha256"] == run.CHECKPOINT_SHA256
    assert all(case["split"] == "train" and case["seen_during_training"] for case in plan["cases"])
    for case in plan["cases"]:
        messages = run.preparation.model_messages(plan["contract"], case)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[1]["content"] == case["task"]


def test_existing_outputs_and_launch_are_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUN", tmp_path)
    with pytest.raises(FileExistsError):
        run.prepare()
    run.diagnostic.save(tmp_path / "diagnostic-launch.json", {"existing": True})
    with pytest.raises(FileExistsError):
        run.supervise()


def test_default_memory_protection_is_not_silently_waived(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUN", tmp_path)
    run.diagnostic.save(tmp_path / "prepared.json", {})
    monkeypatch.setattr(run.training.safety, "resources", lambda: {"c_free_bytes": 9 * run.training.safety.GIB})
    with pytest.raises(RuntimeError, match="resource gate"):
        run.supervise()
    assert not (tmp_path / "diagnostic-launch.json").exists()


def test_real_012_diagnostic_results_and_comparison_are_consistent(monkeypatch):
    if not (run.RUN / "summary.json").exists():
        evidence = run.diagnostic.read(run.RUN / "interruption.json")
        assert evidence["status"] == "interrupted_before_any_recorded_case_result"
        assert evidence["recorded_model_case_result_count"] == evidence["recorded_generation_turn_count"] == 0
        assert evidence["model_task_failure_count"] is None
        assert not evidence["diagnostic_process_found_after_exit"]
        assert not any((run.RUN / name).exists() for name in ("case-results.jsonl", "generation-log.jsonl", "comparison.json"))
        monkeypatch.setattr(run.diagnostic, "RUN", run.RUN)
        preserved = run.diagnostic.read(run.RUN / "input-preservation.json")
        assert preserved["unchanged"] and preserved["all_14_model_workspaces_unmodified"]
        assert not preserved["model_diagnostic_completed"]
        assert run.diagnostic.verify_preservation() == preserved["protected_file_count"]
        return
    from test_stage1_mix011_train_fit_results import test_all_14_real_reports_match_prespecified_train_cases_and_limits
    monkeypatch.setattr(run.diagnostic, "RUN", run.RUN)
    test_all_14_real_reports_match_prespecified_train_cases_and_limits()
    summary = run.diagnostic.read(run.RUN / "summary.json")
    comparison = run.diagnostic.read(run.RUN / "comparison.json")
    assert comparison["strict_delta"] == summary["strict_success"]
    assert comparison["behavior_delta"] == summary["behavior_pass"]
    assert len(comparison["pairs"]) == 14 and comparison["split"] == "train"
    assert comparison["baseline"]["strict_success"] == comparison["baseline"]["behavior_pass"] == 0
    assert summary["checkpoint_id"] == run.TRAIN_ID
    attempt = run.diagnostic.read(run.RUN / "diagnostic-supervisor-attempt.json")
    assert attempt["memory_guard_override"] and attempt["start_limits"] == {"c_free_bytes": 8 * run.training.safety.GIB}
    result = run.diagnostic.read(run.RUN / "diagnostic-supervisor-result.json")
    assert result["exit_code"] == 0 and result["resource_abort"] is None and not result["model_restarted"]
    assert run.diagnostic.verify_preservation() == run.diagnostic.read(run.RUN / "input-preservation.json")["protected_file_count"]
