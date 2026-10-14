"""Additive, one-shot recovery of 012 after verified F: paging activation."""

import os
import signal
import subprocess
import sys
import threading
import time

import run_stage1_mix012_train_fit_diagnostic as candidate

diagnostic = candidate.diagnostic
ROOT = candidate.ROOT
ORIGINAL = candidate.RUN
RUN = ORIGINAL / "recovery-20261011"
safety = candidate.training.safety
START_LIMITS = dict(safety.START_LIMITS)
STOP_LIMITS = dict(safety.STOP_LIMITS)


def configure():
    candidate.RUN = RUN
    candidate.configure()


def verify_original():
    evidence = diagnostic.read(ORIGINAL / "interruption.json")
    if (evidence["recorded_model_case_result_count"] != 0
            or evidence["recorded_generation_turn_count"] != 0
            or evidence["diagnostic_process_found_after_exit"]):
        raise ValueError("Original interruption is not unscored/exited")
    if any((ORIGINAL / name).exists() for name in (
            "summary.json", "case-results.jsonl", "generation-log.jsonl", "comparison.json")):
        raise ValueError("Original attempt has recorded model output")
    diagnostic.RUN = ORIGINAL
    diagnostic.verify_preservation()
    old = diagnostic.read(ORIGINAL / "diagnostic-plan.json")
    for case in old["cases"]:
        if diagnostic.inspect_fixture(case["fixture_id"], ORIGINAL / "workspaces" / case["task_id"]).revision != case["fixture_revision"]:
            raise ValueError("Original interrupted workspace changed")
    configure()
    plan = candidate.build_plan()
    if any(plan[key] != old[key] for key in candidate.SAME_FIELDS):
        raise ValueError("Recovery protocol differs from interrupted attempt")


def paging_state():
    import json
    command = "Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize | ConvertTo-Json -Compress"
    result = subprocess.run([
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "-NoProfile", "-NonInteractive", "-Command", command,
    ], capture_output=True, text=True, timeout=20, check=True)
    rows = json.loads(result.stdout.lstrip("\ufeff").strip())
    rows = rows if isinstance(rows, list) else [rows]
    if not any(row["Name"].lower() == "f:\\pagefile.sys" and row["AllocatedBaseSize"] >= 16384 for row in rows):
        raise ValueError("F: paging is not actually active at 16 GiB or larger")
    return rows


def prepare():
    if RUN.exists():
        raise FileExistsError("Recovery already exists; refusing overwrite")
    verify_original()
    paging = paging_state()
    snapshot = safety.resources()
    failures = safety.below_limits(snapshot, START_LIMITS)
    if failures:
        raise RuntimeError(f"Recovery startup gate failed: {failures}; {snapshot}")
    candidate.prepare()
    diagnostic.save(RUN / "recovery-authorization.json", {
        "basis": "User explicitly authorized a new recovery directory, disposable copies, tests and both planning files",
        "original_attempt_preserved": True, "original_attempt_recorded_cases": 0,
        "original_interruption_sha256": diagnostic.digest(ORIGINAL / "interruption.json"),
        "new_training_authorized": False, "memory_guard_override": False,
        "disk_guard_retained": True, "start_limits": START_LIMITS, "stop_limits": STOP_LIMITS,
        "active_pagefiles": paging, "resources": snapshot,
        "recorded_case_repetitions": 1, "automatic_retry_authorized": False,
    })


def supervise():
    configure()
    if not (RUN / "recovery-authorization.json").exists():
        raise ValueError("Prepare and authorize recovery first")
    if any((RUN / name).exists() for name in ("diagnostic-launch.json", "inference-attempt.json")):
        raise FileExistsError("Recovery inference is one-shot")
    paging_state()
    snapshot = safety.resources()
    if safety.below_limits(snapshot, START_LIMITS):
        raise RuntimeError(f"Recovery resource gate failed: {snapshot}")
    diagnostic.verify_preservation()
    diagnostic.save(RUN / "diagnostic-launch.json", {"started_at": safety.timestamp(),
        "one_shot": True, "memory_guard_override": False, "disk_guard_retained": True,
        "start_limits": START_LIMITS, "stop_limits": STOP_LIMITS, "resources": snapshot})
    stopped = None
    with (RUN / "process.log").open("x", encoding="utf-8") as logfile:
        process = subprocess.Popen([sys.executable, __file__, "--worker"],
            start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        def forward():
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=forward, daemon=True)
        reader.start()
        while process.poll() is None:
            try:
                snapshot = safety.resources()
                diagnostic.append(RUN / "resource-monitor.jsonl", snapshot)
                low = safety.below_limits(snapshot, STOP_LIMITS)
                if low:
                    stopped = {"reason": "Resource threshold crossed", "fields": low, "snapshot": snapshot}
            except Exception as error:
                stopped = {"reason": "Resource monitor unavailable", "error": str(error)}
            if stopped:
                if process.poll() is None:
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
    diagnostic.save(RUN / "supervisor-result.json", {"exit_code": code,
        "resource_abort": stopped, "model_restarted": False, "completed_at": safety.timestamp()})
    if code or stopped:
        raise RuntimeError(f"Recovery interrupted: exit={code}; {stopped}")


def finalize():
    configure()
    end = diagnostic.read(RUN / "supervisor-result.json")
    if end["exit_code"] != 0 or end["resource_abort"] is not None:
        raise ValueError("Recovery did not complete cleanly")
    plan = diagnostic.read(RUN / "diagnostic-plan.json")
    reports = [diagnostic.read(RUN / "case-reports" / (case["task_id"] + ".json")) for case in plan["cases"]]
    summary = diagnostic.read(RUN / "summary.json")
    if len(reports) != 14 or any(summary[key] != value for key, value in diagnostic.summarize(reports).items()):
        raise ValueError("Actual reports do not match summary")
    count = diagnostic.verify_preservation()
    verify_original()
    samples = diagnostic.preparation.records(RUN / "resource-monitor.jsonl")
    if not samples or any(safety.below_limits(row, STOP_LIMITS) for row in samples):
        raise ValueError("Resource samples do not support clean completion")
    diagnostic.save(RUN / "resource-summary.json", {"sample_count": len(samples),
        "minimum_observed_bytes": {key: min(row[key] for row in samples) for key in STOP_LIMITS},
        "memory_guard_override": False, "disk_guard_retained": True,
        "sampling_limit": "Approximately three seconds plus probe latency; transient peaks may be missed"})
    diagnostic.save(RUN / "planning-update.json", {"explicitly_authorized": True,
        "before": diagnostic.read(RUN / "planning-before.json"),
        "after": {p: diagnostic.digest(ROOT / p) for p in diagnostic.DOCS},
        "reason": "Authorized additive 012 diagnostic recovery after F: paging activation; record measured seen-task comparison"})
    diagnostic.save(RUN / "completion.json", {"case_count": 14, "protected_file_count": count,
        "original_interrupted_attempt_preserved": True, "recorded_case_repetitions": 1,
        "checkpoint_promoted": False, "new_training_started": False, "completed_at": safety.timestamp()})


if __name__ == "__main__":
    options = {"--prepare-only": prepare, "--evaluate": supervise, "--finalize": finalize}
    if sys.argv[1:] == ["--worker"]:
        configure()
        candidate.worker()
    elif len(sys.argv) == 2 and sys.argv[1] in options:
        options[sys.argv[1]]()
    else:
        raise ValueError("Use --prepare-only, --evaluate or --finalize")
