"""One-shot 012 train-fit control using the unchanged 011 case runner and grader."""

import json
import shutil
import sys

import run_stage1_mix011_train_fit_diagnostic as diagnostic
import train_stage1_mix012_56step as training

ROOT = diagnostic.ROOT
TRAIN_ID = training.RUN_ID
CHECKPOINT = training.RUN / "checkpoint.json"
CHECKPOINT_SHA256 = "60bf2bd64456b1264cc16dfe425cf7cd9d8ae781853eb6cff83587ebe72f2921"
RUN = ROOT / "runs/evaluation" / (TRAIN_ID + "-train-fit-diagnostic-v1")
BASELINE = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-011-mix006007-28step-train-fit-diagnostic-v1/recovery-20261009"
preparation = diagnostic.preparation
ORIGINAL_BUILD = preparation.build_plan
ORIGINAL_CHECKPOINT = preparation.CHECKPOINT
ORIGINAL_PINS = dict(preparation.PINS)
SAME_FIELDS = ("cases", "contract", "contract_sha256", "case_count", "repetitions", "max_new_tokens",
    "max_tool_calls", "max_generated_turns", "do_sample", "enable_thinking", "grading")
IGNORE_MEMORY_GUARD = False


def configure():
    diagnostic.RUN = RUN
    preparation.RUN_ID = TRAIN_ID
    preparation.CHECKPOINT = CHECKPOINT
    preparation.PROPOSED_OUTPUT = RUN
    preparation.PINS = {**{p: d for p, d in ORIGINAL_PINS.items() if p != ORIGINAL_CHECKPOINT},
        CHECKPOINT: CHECKPOINT_SHA256}
    preparation.build_plan = build_plan


def build_plan():
    configure()
    if diagnostic.digest(ORIGINAL_CHECKPOINT) != ORIGINAL_PINS[ORIGINAL_CHECKPOINT]:
        raise ValueError("Prespecified 011 checkpoint changed")
    plan = ORIGINAL_BUILD()
    old = diagnostic.read(BASELINE / "diagnostic-plan.json")
    if any(plan[key] != old[key] for key in SAME_FIELDS):
        raise ValueError("Cases, prompts, contract, grading or inference budgets differ from 011")
    old_reports = [diagnostic.read(BASELINE / "case-reports" / (case["task_id"] + ".json")) for case in old["cases"]]
    summary = diagnostic.read(BASELINE / "summary.json")
    if any(summary[key] != value for key, value in diagnostic.summarize(old_reports).items()):
        raise ValueError("011 summary differs from actual case reports")
    if summary["strict_success"] != 0 or summary["behavior_pass"] != 0 or summary["case_count"] != 14:
        raise ValueError("Prespecified 011 baseline is not 0/14")
    reload = diagnostic.read(training.RUN / "adapter-reload-validation.json")
    if (plan["checkpoint_step"] != 56 or not reload["independent_process"]
            or not reload["adapter_hashes_verified"] or reload["active_adapters"] != ["default"]):
        raise ValueError("012 checkpoint/reload prerequisites not met")
    plan.update({"baseline_summary_sha256": diagnostic.digest(BASELINE / "summary.json"),
        "baseline_plan_sha256": diagnostic.digest(BASELINE / "diagnostic-plan.json"),
        "baseline_run": BASELINE.relative_to(ROOT).as_posix(),
        "planning_updates": {name: "Record actual authorized 012 seen-task fit comparison without promotion" for name in diagnostic.DOCS},
        "prerequisite_for_execution": "Authorized fourteen train cases; explicit per-invocation memory override permitted"})
    return plan


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite or resume diagnostic")
    plan = build_plan()
    before = diagnostic.protected_hashes()
    audit = ROOT / "tests/literal006-fstring007-mix-sft-audit.json"
    if audit.exists():
        before[audit.relative_to(ROOT).as_posix()] = diagnostic.digest(audit)
    original = diagnostic.read(BASELINE / "prepared.json")
    RUN.mkdir(parents=True)
    diagnostic.save(RUN / "protected-input-hashes.json", before)
    diagnostic.save(RUN / "planning-before.json", {p: diagnostic.digest(ROOT / p) for p in diagnostic.DOCS})
    diagnostic.save(RUN / "diagnostic-plan.json", plan)
    diagnostic.save(RUN / "execution-authorization.json", {"model_diagnostic_authorized": True,
        "training_authorized": False, "old_artifact_edits_authorized": False,
        "scope": "012 fourteen original train tasks once each, unchanged 011 runner and grading, disposable copies",
        "memory_override_requires_explicit_flag": True, "disk_protection_required": True})
    for case in plan["cases"]:
        task_id = case["task_id"]
        diagnostic.copy_source(case, RUN / "workspaces" / task_id)
        for group in ("original", "reference"):
            source = BASELINE / "preflight" / group / task_id
            if group == "original":
                if diagnostic.inspect_fixture(case["fixture_id"], source).revision != case["fixture_revision"]:
                    raise ValueError("Reused original preflight fixture changed")
            elif diagnostic.digest(source / "greeting.py") != original["reference_hashes"][task_id]:
                raise ValueError("Reused reference fixture changed")
            if diagnostic.digest(source / "test_greeting.py") != case["test_sha256"]:
                raise ValueError("Reused preflight test changed")
            shutil.copytree(source, RUN / "preflight" / group / task_id,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        report = diagnostic.read(BASELINE / "preflight/reports" / (task_id + ".json"))
        if report["original"]["exit_code"] != 1 or report["reference"]["exit_code"] != 0:
            raise ValueError("Original/reference preflight receipt does not fail/pass")
        diagnostic.save(RUN / "preflight/reports" / (task_id + ".json"), report)
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "prepared.json", {"case_count": 14, "originals_failed": 14, "references_passed": 14,
        "reference_hashes": original["reference_hashes"], "preflights_reexecuted": False,
        "preflights_reused_from": BASELINE.relative_to(ROOT).as_posix(),
        "diagnostic_plan_sha256": diagnostic.digest(RUN / "diagnostic-plan.json"), "protected_file_count": count})
    print(f"Prepared 012: fourteen unchanged train cases, verified preflight receipts reused, {count} old files protected", flush=True)


def worker():
    configure()
    diagnostic.evaluate()
    old = diagnostic.read(BASELINE / "summary.json")
    new = diagnostic.read(RUN / "summary.json")
    old_plan = diagnostic.read(BASELINE / "diagnostic-plan.json")
    pairs = []
    for case in old_plan["cases"]:
        name = case["task_id"] + ".json"
        prior = diagnostic.read(BASELINE / "case-reports" / name)
        current = diagnostic.read(RUN / "case-reports" / name)
        pairs.append({"task_id": case["task_id"], "strict_011": prior["strict_success"],
            "strict_012": current["strict_success"], "behavior_011": prior["behavior_pass"],
            "behavior_012": current["behavior_pass"], "source_changed_012": current["source_changed"],
            "denied_011": prior["denied_command_count"], "denied_012": current["denied_command_count"]})
    diagnostic.save(RUN / "comparison.json", {"split": "train", "baseline": diagnostic.summarize([
        diagnostic.read(BASELINE / "case-reports" / (case["task_id"] + ".json")) for case in old_plan["cases"]]),
        "candidate": {key: new[key] for key in diagnostic.summarize([])},
        "strict_delta": new["strict_success"] - old["strict_success"],
        "behavior_delta": new["behavior_pass"] - old["behavior_pass"], "pairs": pairs,
        "interpretation": "Same-SFT budget comparison; linear LR horizon also changed. Seen tasks only, no unseen-generalization or isolated causal claim."})


def supervise():
    if (RUN / "diagnostic-launch.json").exists() or (RUN / "inference-attempt.json").exists():
        raise FileExistsError("Refusing repeated diagnostic inference")
    if not (RUN / "prepared.json").exists():
        raise ValueError("Prepare the new diagnostic first")
    training.RUN = RUN
    training.__file__ = __file__
    training.IGNORE_MEMORY_GUARD = IGNORE_MEMORY_GUARD
    snapshot = training.safety.resources()
    if training.safety.below_limits(snapshot, training.resource_limits(True)):
        raise RuntimeError(f"Diagnostic resource gate not met: {snapshot}")
    diagnostic.save(RUN / "diagnostic-launch.json", {"one_shot": True,
        "started_at": training.safety.timestamp(), "resources": snapshot,
        "memory_guard_override": IGNORE_MEMORY_GUARD, "disk_guard_retained": True,
        "basis": "User separately authorized fourteen train tasks and this invocation's memory-only override"})
    training.supervise("diagnostic")


def record_interruption():
    configure()
    evidence = diagnostic.read(RUN / "interruption.json")
    if evidence["recorded_model_case_result_count"] != 0 or evidence["diagnostic_process_found_after_exit"]:
        raise ValueError("Interruption evidence is inconsistent")
    if any((RUN / name).exists() for name in ("summary.json", "comparison.json", "case-results.jsonl", "generation-log.jsonl")):
        raise ValueError("Recorded model output exists; do not describe an unscored interruption")
    plan = diagnostic.read(RUN / "diagnostic-plan.json")
    for case in plan["cases"]:
        workspace = RUN / "workspaces" / case["task_id"]
        if diagnostic.inspect_fixture(case["fixture_id"], workspace).revision != case["fixture_revision"]:
            raise ValueError("Interrupted workspace changed; manual review required")
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "input-preservation.json", {"unchanged": True,
        "protected_file_count": count, "changed_paths": [], "all_14_model_workspaces_unmodified": True,
        "model_diagnostic_completed": False, "model_task_failure_count": None})
    print(f"Recorded unscored interruption; {count} protected files and fourteen model copies unchanged", flush=True)


def main():
    global IGNORE_MEMORY_GUARD
    args = sys.argv[1:]
    if "--ignore-memory-guard" in args:
        IGNORE_MEMORY_GUARD = True
        args.remove("--ignore-memory-guard")
    if args == ["--prepare-only"]:
        prepare()
    elif args == ["--evaluate"]:
        supervise()
    elif args == ["--diagnostic-worker"]:
        worker()
    elif args == ["--record-interruption"]:
        record_interruption()
    elif args == ["--check"]:
        print(json.dumps(build_plan(), indent=2), flush=True)
    else:
        raise ValueError("Use --prepare-only, --evaluate or --check; optional --ignore-memory-guard")


if __name__ == "__main__":
    main()
