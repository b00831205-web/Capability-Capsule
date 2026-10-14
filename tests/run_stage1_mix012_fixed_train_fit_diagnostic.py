"""One-shot, train-only 012 probe using opt-in literal/termination evaluator fixes."""

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import run_stage1_mix012_train_fit_diagnostic as candidate
from capability_capsule.eval import coding_checkpoint_v2 as fixed

diagnostic = candidate.diagnostic
ROOT = candidate.ROOT
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-evalfix-v2"
PRIOR = candidate.RUN / "recovery-20261011"
ORIGINAL_BUILD = candidate.build_plan
safety = candidate.training.safety
FIXED_PATH = ROOT / "src/capability_capsule/eval/coding_checkpoint_v2.py"
FIXED_SHA256 = "61d3c70099300761d263f7e0d6d97b6a179534bed6cc01fea84b3f755e5c5106"
MARKUP = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")


def configure():
    candidate.RUN = RUN
    candidate.build_plan = build_plan
    candidate.configure()


def build_plan():
    plan = ORIGINAL_BUILD()
    old = diagnostic.read(PRIOR / "diagnostic-plan.json")
    if any(plan[k] != old[k] for k in candidate.SAME_FIELDS):
        raise ValueError("Original 012 tasks/contract/budgets/grading changed")
    if diagnostic.digest(FIXED_PATH) != FIXED_SHA256:
        raise ValueError("Fixed evaluator implementation changed")
    plan.update({"diagnostic_id": RUN.name, "proposed_output": RUN.relative_to(ROOT).as_posix(),
        "implementation_id": fixed.IMPLEMENTATION_ID,
        "implementation_sha256": FIXED_SHA256,
        "runner_sha256": diagnostic.digest(Path(__file__)),
        "comparison_run": PRIOR.relative_to(ROOT).as_posix(),
        "comparison_plan_sha256": diagnostic.digest(PRIOR / "diagnostic-plan.json"),
        "comparison_summary_sha256": diagnostic.digest(PRIOR / "summary.json"),
        "scope": "Seen train tasks only; executor/termination control, not validation or training",
        "changed_factors": ["literal-safe executor", "actual EOS/length/unknown detection", "separate format telemetry"],
        "recording": "Independent train-only diagnostic; no validation ledger or checkpoint promotion"})
    return plan


def prepare():
    configure()
    if RUN.exists():
        raise FileExistsError("Refusing overwrite or resume")
    plan = build_plan()
    before = diagnostic.protected_hashes()
    original = diagnostic.read(PRIOR / "prepared.json")
    RUN.mkdir(parents=True)
    diagnostic.save(RUN / "protected-input-hashes.json", before)
    diagnostic.save(RUN / "planning-before.json", {p:diagnostic.digest(ROOT / p) for p in diagnostic.DOCS})
    diagnostic.save(RUN / "diagnostic-plan.json", plan)
    for case in plan["cases"]:
        task_id = case["task_id"]
        diagnostic.copy_source(case, RUN / "workspaces" / task_id)
        for group in ("original", "reference"):
            source = PRIOR / "preflight" / group / task_id
            if diagnostic.digest(source / "test_greeting.py") != case["test_sha256"]:
                raise ValueError("Preflight test changed")
            if group == "original":
                if diagnostic.inspect_fixture(case["fixture_id"], source).revision != case["fixture_revision"]:
                    raise ValueError("Preflight original changed")
            elif diagnostic.digest(source / "greeting.py") != original["reference_hashes"][task_id]:
                raise ValueError("Preflight reference changed")
            shutil.copytree(source, RUN / "preflight" / group / task_id,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        report = diagnostic.read(PRIOR / "preflight/reports" / (task_id + ".json"))
        if report["original"]["exit_code"] != 1 or report["reference"]["exit_code"] != 0:
            raise ValueError("Preflight receipt does not fail/pass")
        diagnostic.save(RUN / "preflight/reports" / (task_id + ".json"), report)
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "prepared.json", {"case_count":14, "originals_failed":14, "references_passed":14,
        "reference_hashes":original["reference_hashes"], "preflights_reexecuted":False,
        "preflights_reused_from":PRIOR.relative_to(ROOT).as_posix(),
        "diagnostic_plan_sha256":diagnostic.digest(RUN / "diagnostic-plan.json"), "protected_file_count":count})
    diagnostic.save(RUN / "fixed-diagnostic-authorization.json", {
        "basis": "User authorized same fourteen seen tasks with repaired evaluator, unchanged budget, new outputs only",
        "model_diagnostic_authorized": True, "training_authorized": False,
        "data_changes_authorized": False, "old_results_overwritten": False,
        "memory_guard_override": False, "disk_guard_retained": True,
        "automatic_retry_authorized": False, "recorded_case_repetitions": 1,
        "start_limits": safety.START_LIMITS, "stop_limits": safety.STOP_LIMITS})
    print(f"Prepared fourteen unchanged train tasks; {count} protected files", flush=True)


def run_case(case, contract, workspace, generator, executor, expected_source, on_turn=lambda row: None):
    if case["split"] != "train":
        raise ValueError("Train-only diagnostic")
    started = time.monotonic()
    initial_source = diagnostic.digest(workspace / "greeting.py")
    initial_test = diagnostic.digest(workspace / "test_greeting.py")
    messages = candidate.preparation.model_messages(contract, case)
    turns, executions = [], []
    reached_final, error, reason = False, None, "error"
    try:
        for index in range(5):
            generated = fixed.GeneratedTurn.model_validate(generator.generate(messages, contract["tools"]).model_dump())
            turn = {"turn": index + 1, **generated.model_dump(mode="json"), "format_error": None}
            turns.append(turn)
            parsed = None
            try:
                parsed = diagnostic.parse_qwen_tool_completion(generated.completion)
                if any(marker in parsed.content for marker in MARKUP):
                    raise ValueError("Unmatched tool markup outside the parsed call")
            except ValueError as exc:
                turn["format_error"] = str(exc)
            if generated.finish_reason != "eos":
                reason = "generation_length" if generated.finish_reason == "length" else "unknown_finish"
                error = reason
                on_turn(turn)
                break
            if turn["format_error"] is not None:
                reason = "malformed_tool_call" if any(m in generated.completion for m in MARKUP) else "error"
                error = turn["format_error"]
                on_turn(turn)
                break
            if not parsed.tool_calls:
                reached_final, reason = True, "final_response"
                on_turn(turn)
                break
            if len(executions) >= 4:
                reason, error = "tool_budget", "ToolRoundLimitExceeded"
                on_turn(turn)
                break
            call = parsed.tool_calls[0]
            if call.name != "exec_command" or not isinstance(call.arguments.get("cmd"), str):
                turn["format_error"] = "Invalid exec_command request"
                on_turn(turn)
                reason = "malformed_tool_call"
                raise ValueError("Invalid exec_command request")
            command = call.arguments["cmd"]
            execution = executor.execute(command, workspace)
            entry = {"command": command, **execution.model_dump(mode="json")}
            executions.append(entry)
            turn["execution"] = entry
            on_turn(turn)
            messages.extend([
                {"role": "assistant", "content": parsed.content, "tool_calls": [
                    {"type": "function", "function": {"name": call.name, "arguments": dict(call.arguments)}}]},
                {"role": "tool", "name": call.name, "content": execution.envelope()}])
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        on_turn({"error": error})
    tests_unchanged = diagnostic.digest(workspace / "test_greeting.py") == initial_test
    behavior = diagnostic.pytest_result(workspace) if tests_unchanged else {"exit_code": 125, "output": "Test bytes changed"}
    exact = (workspace / "greeting.py").read_text("utf-8") == expected_source
    denied = sum(not row["authorized"] for row in executions)
    strict = reached_final and error is None and tests_unchanged and exact and behavior["exit_code"] == 0 and denied == 0
    duration = (time.monotonic() - started) * 1000
    return {"task_id": case["task_id"], "split": "train", "seen_during_training": True,
        "implementation_id": fixed.IMPLEMENTATION_ID, "strict_success": strict,
        "behavior_pass": behavior["exit_code"] == 0, "exact_source_pass": exact,
        "tests_unchanged": tests_unchanged, "source_changed": diagnostic.digest(workspace / "greeting.py") != initial_source,
        "model_invoked_pytest": any(r["authorized"] and r["command"].strip().lower() in {
            "pytest -q", "pytest -q test_greeting.py", "python -m pytest -q", "python -m pytest -q test_greeting.py"} for r in executions),
        "denied_command_count": denied, "tool_call_count": len(executions), "generation_turn_count": len(turns),
        "reached_final": reached_final, "error": error, "termination_reason": reason,
        "format_error_count": sum(t["format_error"] is not None for t in turns),
        "truncated_turn_count": sum(t["finish_reason"] == "length" for t in turns),
        "edit_attempt_count": sum(".replace(" in r["command"].lower() for r in executions),
        "failed_execution_count": sum(r["exit_code"] != 0 for r in executions),
        "final_edit_incorrect": not exact, "duration_ms": duration,
        "time_bounded_success": strict and duration <= 180000,
        "turns": turns, "tool_results": executions, "independent_pytest": behavior}


def worker():
    configure()
    prepared = diagnostic.read(RUN / "prepared.json")
    plan = diagnostic.read(RUN / "diagnostic-plan.json")
    if diagnostic.digest(RUN / "diagnostic-plan.json") != prepared["diagnostic_plan_sha256"]:
        raise ValueError("Prepared plan changed")
    current = build_plan()
    if any(current[k] != plan[k] for k in ("input_hashes", "implementation_sha256", "runner_sha256",
            "comparison_plan_sha256", "comparison_summary_sha256")):
        raise ValueError("Input/implementation hashes changed")
    diagnostic.verify_preservation()
    for case in plan["cases"]:
        path = RUN / "workspaces" / case["task_id"]
        if diagnostic.inspect_fixture(case["fixture_id"], path).revision != case["fixture_revision"]:
            raise ValueError("Model workspace already changed")
        reference = RUN / "preflight/reference" / case["task_id"] / "greeting.py"
        if diagnostic.digest(reference) != prepared["reference_hashes"][case["task_id"]]:
            raise ValueError("Reference changed")
    diagnostic.save(RUN / "inference-attempt.json", {"one_shot": True, "started_at": safety.timestamp(),
        "checkpoint_sha256": candidate.CHECKPOINT_SHA256, "implementation_sha256": FIXED_SHA256})
    generator = diagnostic.ProgressGenerator(fixed.TransformersPeftTurnGenerator(candidate.CHECKPOINT, max_new_tokens=256))
    reports = []
    pairs = []
    for case in plan["cases"]:
        task_id = case["task_id"]
        print(f"Starting {len(reports)+1}/14: {task_id}", flush=True)
        expected = (RUN / "preflight/reference" / task_id / "greeting.py").read_text("utf-8")
        report = run_case(case, plan["contract"], RUN / "workspaces" / task_id, generator,
            fixed.ConstrainedPowerShellExecutor(), expected,
            on_turn=lambda row: diagnostic.append(RUN / "generation-log.jsonl", {"task_id": task_id, **row}))
        diagnostic.save(RUN / "case-reports" / (task_id + ".json"), report)
        diagnostic.append(RUN / "case-results.jsonl", {k:v for k,v in report.items() if k not in {"turns", "tool_results", "independent_pytest"}})
        reports.append(report)
        old = diagnostic.read(PRIOR / "case-reports" / (task_id + ".json"))
        pairs.append({"task_id": task_id, "old_strict": old["strict_success"], "new_strict": report["strict_success"],
            "old_behavior": old["behavior_pass"], "new_behavior": report["behavior_pass"],
            "old_denied": old["denied_command_count"], "new_denied": report["denied_command_count"],
            "new_termination": report["termination_reason"]})
        print(f"strict={report['strict_success']}, behavior={report['behavior_pass']}, stop={report['termination_reason']}", flush=True)
    summary = {**diagnostic.summarize(reports), "split": "train", "seen_during_training": True,
        "groups": {"literal006": diagnostic.summarize(reports[:8]), "fstring007": diagnostic.summarize(reports[8:])},
        "termination_reasons": dict(Counter(r["termination_reason"] for r in reports)),
        **{key:sum(r[key] for r in reports) for key in ("format_error_count", "truncated_turn_count", "final_edit_incorrect", "edit_attempt_count", "failed_execution_count")},
        "completed_at": safety.timestamp(), "checkpoint_promoted": False,
        "interpretation": "Same checkpoint and seen tasks; executor/termination semantics changed. Not training improvement or unseen validation."}
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "summary.json", summary)
    diagnostic.save(RUN / "comparison.json", {"baseline": diagnostic.read(PRIOR / "summary.json"), "candidate": summary, "pairs": pairs})
    diagnostic.save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count, "changed_paths": []})
    print(json.dumps(summary, indent=2), flush=True)


def supervise():
    configure()
    if not (RUN / "fixed-diagnostic-authorization.json").exists():
        raise ValueError("Prepare first")
    if any((RUN / name).exists() for name in ("diagnostic-launch.json", "inference-attempt.json")):
        raise FileExistsError("One-shot inference; no automatic retry")
    snapshot = safety.resources()
    low = safety.below_limits(snapshot, safety.START_LIMITS)
    if low:
        raise RuntimeError(f"Startup resource protection: {low}; {snapshot}")
    diagnostic.verify_preservation()
    diagnostic.save(RUN / "diagnostic-launch.json", {"started_at": safety.timestamp(), "resources": snapshot,
        "memory_guard_override": False, "start_limits": safety.START_LIMITS, "stop_limits": safety.STOP_LIMITS})
    stopped = None
    with (RUN / "process.log").open("x", encoding="utf-8") as logfile:
        process = subprocess.Popen([sys.executable, __file__, "--worker"], start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        def forward():
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)
        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        while process.poll() is None:
            try:
                sample = safety.resources()
                diagnostic.append(RUN / "resource-monitor.jsonl", sample)
                low = safety.below_limits(sample, safety.STOP_LIMITS)
                if low:
                    stopped = {"fields": low, "snapshot": sample}
            except Exception as exc:
                stopped = {"probe_error": str(exc)}
            if stopped:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(3)
        code = process.wait()
        reader.join(timeout=10)
    diagnostic.save(RUN / "supervisor-result.json", {"exit_code": code, "resource_abort": stopped,
        "model_restarted": False, "completed_at": safety.timestamp()})
    if code or stopped:
        raise RuntimeError(f"Diagnostic interrupted: exit={code}; {stopped}")


def finalize():
    configure()
    end = diagnostic.read(RUN / "supervisor-result.json")
    if end["exit_code"] or end["resource_abort"] is not None:
        raise ValueError("Run did not finish cleanly")
    plan = diagnostic.read(RUN / "diagnostic-plan.json")
    reports = [diagnostic.read(RUN / "case-reports" / (c["task_id"] + ".json")) for c in plan["cases"]]
    summary = diagnostic.read(RUN / "summary.json")
    if len(reports) != 14 or any(summary[k] != v for k,v in diagnostic.summarize(reports).items()):
        raise ValueError("Reports/summary mismatch")
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "planning-update.json", {"explicitly_authorized": True,
        "before": diagnostic.read(RUN / "planning-before.json"),
        "after": {p:diagnostic.digest(ROOT / p) for p in diagnostic.DOCS},
        "reason": "Authorized same-task repaired-evaluator 012 diagnostic; preserve all historical results"})
    diagnostic.save(RUN / "completion.json", {"case_count":14, "protected_file_count":count,
        "new_training_started":False, "checkpoint_promoted":False, "completed_at":safety.timestamp()})


if __name__ == "__main__":
    options = {"--prepare-only":prepare, "--evaluate":supervise, "--worker":worker, "--finalize":finalize}
    if len(sys.argv) != 2 or sys.argv[1] not in options:
        raise ValueError("Use --prepare-only, --evaluate, --worker or --finalize")
    options[sys.argv[1]]()
