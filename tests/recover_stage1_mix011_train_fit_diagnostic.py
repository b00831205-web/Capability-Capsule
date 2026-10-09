"""Recover the authorized interrupted train probe, retaining every original receipt."""

import json
import ctypes
import gc
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import run_stage1_mix011_train_fit_diagnostic as diagnostic

ROOT = diagnostic.ROOT
ORIGINAL = diagnostic.RUN
RUN = ORIGINAL / "recovery-20261009"
GIB = 1024 ** 3
START_LIMITS = {"c_free_bytes": 8 * GIB, "physical_free_bytes": 10 * GIB,
                "commit_free_bytes": 14 * GIB, "wsl_available_bytes": 10 * GIB}
STOP_LIMITS = {"c_free_bytes": 4 * GIB, "physical_free_bytes": int(1.5 * GIB),
               "commit_free_bytes": 3 * GIB, "wsl_available_bytes": GIB}


def timestamp():
    return datetime.now(UTC).isoformat()


def resources():
    command = (
        "$resourceOs = Get-CimInstance Win32_OperatingSystem; "
        "$resourceDisk = Get-PSDrive -Name C; "
        "@{physical_free_bytes=[long]$resourceOs.FreePhysicalMemory*1024; "
        "commit_free_bytes=[long]$resourceOs.FreeVirtualMemory*1024; "
        "c_free_bytes=[long]$resourceDisk.Free} | ConvertTo-Json -Compress"
    )
    result = subprocess.run([
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "-NoProfile", "-NonInteractive", "-Command", command,
    ], capture_output=True, text=True, timeout=15, check=True)
    data = json.loads(result.stdout.lstrip("\ufeff").strip())
    meminfo = dict(line.split(":", 1) for line in open("/proc/meminfo", encoding="utf-8"))
    data["wsl_available_bytes"] = int(meminfo["MemAvailable"].split()[0]) * 1024
    data["observed_at"] = timestamp()
    return data


def below_limits(snapshot, limits):
    return [key for key, limit in limits.items() if snapshot.get(key, -1) < limit]


def prepare():
    if RUN.exists():
        raise FileExistsError("Recovery already prepared; refusing overwrite")
    if any((ORIGINAL / name).exists() for name in ("case-results.jsonl", "generation-log.jsonl", "summary.json")):
        raise ValueError("Original attempt has model output; cannot treat cases as unexecuted")
    interruption = diagnostic.read(ORIGINAL / "model-load-interruption.json")
    if interruption["recorded_model_case_result_count"] != 0:
        raise ValueError("Original cases already recorded")
    diagnostic.verify_preservation()
    old_plan = diagnostic.read(ORIGINAL / "diagnostic-plan.json")
    current = diagnostic.preparation.build_plan()
    if current["cases"] != old_plan["cases"] or current["contract"] != old_plan["contract"]:
        raise ValueError("Original diagnostic conditions changed")
    old_prepared = diagnostic.read(ORIGINAL / "prepared.json")
    for case in old_plan["cases"]:
        if diagnostic.inspect_fixture(case["fixture_id"], ORIGINAL / "workspaces" / case["task_id"]).revision != case["fixture_revision"]:
            raise ValueError("Original model workspace changed")
        if diagnostic.digest(ORIGINAL / "preflight/reference" / case["task_id"] / "greeting.py") != old_prepared["reference_hashes"][case["task_id"]]:
            raise ValueError("Original reference changed")
    # Release temporary JSON/file buffers before measuring model-load headroom.
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    host = resources()
    failures = below_limits(host, START_LIMITS)
    for _ in range(6):
        if not failures:
            break
        print("Waiting for resource headroom: " + json.dumps(host), flush=True)
        time.sleep(5)
        host = resources()
        failures = below_limits(host, START_LIMITS)
    if failures:
        raise RuntimeError(f"Insufficient resource headroom before recovery: {failures}; {host}")
    diagnostic.RUN = RUN
    before = diagnostic.protected_hashes()
    RUN.mkdir()
    diagnostic.save(RUN / "protected-input-hashes.json", before)
    diagnostic.save(RUN / "planning-before.json", {p: diagnostic.digest(ROOT / p) for p in diagnostic.DOCS})
    plan = {**old_plan, "diagnostic_id": old_plan["diagnostic_id"] + "/recovery-20261009",
            "proposed_output": RUN.relative_to(ROOT).as_posix(), "output_already_exists": False}
    diagnostic.save(RUN / "diagnostic-plan.json", plan)
    for case in plan["cases"]:
        task_id = case["task_id"]
        diagnostic.copy_source(case, RUN / "workspaces" / task_id)
        for group in ("original", "reference"):
            shutil.copytree(ORIGINAL / "preflight" / group / task_id, RUN / "preflight" / group / task_id,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        report = diagnostic.read(ORIGINAL / "preflight/reports" / (task_id + ".json"))
        if report["original"]["exit_code"] != 1 or report["reference"]["exit_code"] != 0:
            raise ValueError("Original preflight evidence failed")
        diagnostic.save(RUN / "preflight/reports" / (task_id + ".json"), report)
    count = diagnostic.verify_preservation()
    diagnostic.save(RUN / "prepared.json", {**old_prepared, "prepared_at": timestamp(),
        "diagnostic_plan_sha256": diagnostic.digest(RUN / "diagnostic-plan.json"),
        "protected_file_count": count, "preflights_reused_from": ORIGINAL.relative_to(ROOT).as_posix(),
        "preflights_reexecuted": False})
    diagnostic.save(RUN / "recovery-authorization.json", {
        "basis": "Existing authorization for fourteen-case diagnostic plus user request to continue after cleanup",
        "original_attempt_preserved": True, "recorded_case_repetitions": 1,
        "new_training_authorized": False, "original_prepared_sha256": diagnostic.digest(ORIGINAL / "prepared.json"),
        "start_limits_bytes": START_LIMITS, "stop_limits_bytes": STOP_LIMITS, "resource_snapshot": host,
    })
    print(f"Recovery prepared: 14 original train cases, {count} old files protected", flush=True)


def supervise():
    if (RUN / "supervisor-attempt.json").exists():
        raise FileExistsError("Supervisor attempt already exists")
    snapshot = resources()
    for _ in range(12):
        if not below_limits(snapshot, START_LIMITS):
            break
        diagnostic.append(RUN / "startup-resource-checks.jsonl", snapshot)
        print("Waiting before model load: " + json.dumps(snapshot), flush=True)
        time.sleep(5)
        snapshot = resources()
    if below_limits(snapshot, START_LIMITS):
        raise RuntimeError("Resource headroom is below the preparation threshold")
    diagnostic.save(RUN / "supervisor-attempt.json", {"started_at": timestamp(), "resources": snapshot})
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
        stopped = None
        while process.poll() is None:
            try:
                snapshot = resources()
                diagnostic.append(RUN / "resource-monitor.jsonl", snapshot)
                low = below_limits(snapshot, STOP_LIMITS)
                if low:
                    stopped = {"reason": "Resource threshold crossed", "fields": low, "snapshot": snapshot}
            except Exception as error:
                stopped = {"reason": "Resource monitor unavailable", "error": str(error)}
            if stopped:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(3)
        exit_code = process.wait()
        reader.join(timeout=10)
    diagnostic.save(RUN / "supervisor-result.json", {"completed_at": timestamp(),
        "exit_code": exit_code, "resource_abort": stopped, "model_restarted": False})
    if exit_code != 0 or stopped:
        raise RuntimeError(f"Recovery did not complete: exit={exit_code}, resource_abort={stopped}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--evaluate"]:
        supervise()
    elif sys.argv[1:] == ["--worker"]:
        diagnostic.RUN = RUN
        diagnostic.evaluate()
    else:
        raise ValueError("Use --prepare-only or --evaluate")
