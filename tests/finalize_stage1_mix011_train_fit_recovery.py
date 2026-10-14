"""Seal a completed recovery's preservation, resource and planning receipts."""

import recover_stage1_mix011_train_fit_diagnostic as recovery


def finalize():
    diagnostic = recovery.diagnostic
    diagnostic.RUN = recovery.RUN
    end = diagnostic.read(recovery.RUN / "supervisor-result.json")
    if end["exit_code"] != 0 or end["resource_abort"] is not None:
        raise ValueError("Only a completed, non-aborted recovery can be finalized")
    plan = diagnostic.read(recovery.RUN / "diagnostic-plan.json")
    reports = [diagnostic.read(recovery.RUN / "case-reports" / (case["task_id"] + ".json")) for case in plan["cases"]]
    if len(reports) != 14 or len({r["task_id"] for r in reports}) != 14:
        raise ValueError("Expected fourteen once-only reports")
    summary = diagnostic.read(recovery.RUN / "summary.json")
    if any(summary[key] != value for key, value in diagnostic.summarize(reports).items()):
        raise ValueError("Summary differs from actual reports")
    for case in plan["cases"]:
        task_id = case["task_id"]
        if diagnostic.digest(recovery.RUN / "workspaces" / task_id / "test_greeting.py") != case["test_sha256"]:
            raise ValueError("Supplied tests changed")
    preserved_count = diagnostic.verify_preservation()
    diagnostic.RUN = recovery.ORIGINAL
    try:
        original_preserved_count = diagnostic.verify_preservation()
    finally:
        diagnostic.RUN = recovery.RUN
    samples = diagnostic.preparation.records(recovery.RUN / "resource-monitor.jsonl")
    if not samples or any(recovery.below_limits(sample, recovery.STOP_LIMITS) for sample in samples):
        raise ValueError("Resource samples inconsistent with completed supervisor")
    diagnostic.save(recovery.RUN / "resource-summary.json", {
        "sample_count": len(samples), "first_sample_at": samples[0]["observed_at"],
        "last_sample_at": samples[-1]["observed_at"],
        "minimum_observed_bytes": {key: min(s[key] for s in samples) for key in recovery.STOP_LIMITS},
        "resource_abort": None, "sampling_limit": "Approximately three seconds plus probe latency; not continuous peak capture",
    })
    diagnostic.save(recovery.RUN / "planning-update.json", {"explicitly_authorized": True,
        "before": diagnostic.read(recovery.RUN / "planning-before.json"),
        "after": {p: diagnostic.digest(recovery.ROOT / p) for p in diagnostic.DOCS},
        "reason": "Continue the authorized interrupted diagnostic after user cleanup and record actual train-fit outcomes"})
    diagnostic.save(recovery.RUN / "completion.json", {
        "completed_at": recovery.timestamp(), "case_count": 14,
        "protected_file_count": preserved_count, "recorded_case_repetitions": 1,
        "original_legacy_protected_file_count": original_preserved_count,
        "original_interrupted_attempt_preserved": True, "original_attempt_recorded_cases": 0,
        "model_case_reruns": 0, "checkpoint_promoted": False, "new_training_started": False,
    })
    print(f"Finalized fourteen train cases; {preserved_count} protected inputs unchanged", flush=True)


if __name__ == "__main__":
    finalize()
