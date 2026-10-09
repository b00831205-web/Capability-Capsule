"""Real diagnostic evidence remains train-only and agrees with every case report."""

import json

import run_stage1_mix011_train_fit_diagnostic as diagnostic


def test_all_14_real_reports_match_prespecified_train_cases_and_limits():
    if not (diagnostic.RUN / "summary.json").exists():
        interruption = diagnostic.read(diagnostic.RUN / "model-load-interruption.json")
        assert interruption["status"] == "interrupted_before_any_recorded_case_result"
        assert interruption["recorded_model_case_result_count"] == interruption["recorded_generation_turn_count"] == 0
        assert interruption["model_task_failure_count"] is None
        assert not (diagnostic.RUN / "case-results.jsonl").exists()
        assert not (diagnostic.RUN / "generation-log.jsonl").exists()
        assert not list((diagnostic.RUN / "case-reports").glob("*.json"))
        return
    plan = diagnostic.read(diagnostic.RUN / "diagnostic-plan.json")
    reports = [diagnostic.read(diagnostic.RUN / "case-reports" / (case["task_id"] + ".json")) for case in plan["cases"]]
    rows = [json.loads(line) for line in (diagnostic.RUN / "case-results.jsonl").read_text("utf-8").splitlines()]
    assert len(reports) == len(rows) == len({r["task_id"] for r in reports}) == 14
    assert [r["task_id"] for r in rows] == [c["task_id"] for c in plan["cases"]]
    for case, report, row in zip(plan["cases"], reports, rows, strict=True):
        assert case["split"] == report["split"] == row["split"] == "train"
        assert report["seen_during_training"]
        assert report["tool_call_count"] == len(report["tool_results"]) <= 4
        assert report["generation_turn_count"] == len(report["turns"]) <= 5
        assert report["denied_command_count"] == sum(not t["authorized"] for t in report["tool_results"])
        assert report["behavior_pass"] == (report["independent_pytest"]["exit_code"] == 0)
        assert report["strict_success"] == (report["reached_final"] and report["error"] is None
            and report["behavior_pass"] and report["exact_source_pass"]
            and report["tests_unchanged"] and report["denied_command_count"] == 0)
        workspace = diagnostic.RUN / "workspaces" / case["task_id"]
        assert diagnostic.digest(workspace / "test_greeting.py") == case["test_sha256"]
        assert diagnostic.digest(diagnostic.ROOT / case["fixture_root"] / "greeting.py") == case["source_sha256"]
    summary = diagnostic.read(diagnostic.RUN / "summary.json")
    for key, expected in diagnostic.summarize(reports).items():
        assert summary[key] == expected
    assert summary["groups"]["literal006"] == diagnostic.summarize(reports[:8])
    assert summary["groups"]["fstring007"] == diagnostic.summarize(reports[8:])
    assert summary["reruns"] == 0 and not summary["checkpoint_promoted"]
    assert not list(diagnostic.RUN.rglob("learning-curve.jsonl"))


def test_preflights_and_old_artifacts_remain_verified():
    prepared = diagnostic.read(diagnostic.RUN / "prepared.json")
    assert prepared["originals_failed"] == prepared["references_passed"] == 14
    for task_id, pinned in prepared["reference_hashes"].items():
        assert diagnostic.digest(diagnostic.RUN / "preflight/reference" / task_id / "greeting.py") == pinned
        result = diagnostic.read(diagnostic.RUN / "preflight/reports" / (task_id + ".json"))
        assert result["original"]["exit_code"] == 1 and result["reference"]["exit_code"] == 0
    preserved = diagnostic.read(diagnostic.RUN / "input-preservation.json")
    assert preserved["unchanged"] and preserved["changed_paths"] == []
    assert diagnostic.verify_preservation() == preserved["protected_file_count"]
    assert diagnostic.read(diagnostic.RUN / "inference-attempt.json")["one_shot"]
    interruption = diagnostic.read(diagnostic.RUN / "preinference-interruption.json")
    assert not interruption["model_inference_started"] and interruption["model_case_reruns"] == 0
