"""Separately requested float32 attempt; retain old run and tighten startup RAM gate."""

from pathlib import Path
import sys

import run_stage1_mix_teacher_forced_diagnostic as forward

ROOT = forward.ROOT
ORIGINAL = forward.RUN
RUN = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1-retry-20261013"
DEPENDENCY = Path(forward.__file__)
DEPENDENCY_SHA256 = "b1f643a3d511f18f0c2ad355d5d68371bffd381b98343845e86f260f68d207d4"
ORIGINAL_PINS = dict(forward.PINS)
START_LIMITS = {**forward.safety.START_LIMITS,"physical_free_bytes":14 * 1024**3}


def configure():
    if forward.digest(DEPENDENCY) != DEPENDENCY_SHA256:
        raise ValueError("Frozen forward implementation changed")
    forward.RUN = RUN
    forward.__file__ = __file__
    forward.PINS = {**ORIGINAL_PINS,DEPENDENCY:DEPENDENCY_SHA256}
    forward.safety.START_LIMITS = dict(START_LIMITS)


def prepare():
    configure()
    original = forward.read(ORIGINAL / "interruption.json")
    if original["recorded_complete_case_count"] != 0 or original["011_started"] or original["012_started"]:
        raise ValueError("Prespecified prior attempt is not the recorded unscored interruption")
    forward.prepare()
    forward.save(RUN / "retry-authorization.json",{
        "basis":"User requested next attempt after memory-guard interruption",
        "prior_attempt":ORIGINAL.relative_to(ROOT).as_posix(),
        "prior_interruption_sha256":forward.digest(ORIGINAL / "interruption.json"),
        "changed_factor":"Startup physical-free-memory minimum raised from 10 to 14 GiB",
        "unchanged":"Gold data, models, float32 CPU forward scoring, categories and runtime guards",
        "memory_guard_override":False,"disk_guard_retained":True,
        "automatic_retry":False,"generation":False,"command_execution":False,"training":False})


def record_status():
    configure()
    if (RUN / "comparison.json").exists():
        forward.finalize()
        return
    cases,events = {},{}
    for variant in forward.VARIANTS:
        path = RUN / variant
        end = path / "supervisor-result.json"
        blocked = RUN / (variant + "-startup-blocked.json")
        if path.exists() and not end.exists():
            raise ValueError("Variant process exit not yet confirmed")
        cases[variant] = len(list((path / "cases").glob("*.json")))
        if end.exists():
            events[variant] = forward.read(end)
        if blocked.exists():
            events[variant] = {"startup_blocked":forward.read(blocked)}
    if not events:
        raise ValueError("No recorded completion or resource interruption")
    count = forward.verify_preservation()
    forward.save(RUN / "interruption.json",{
        "diagnostic_comparison_completed":False,"recorded_case_counts":cases,"variant_events":events,
        "case_success_count":None,"generation_calls":0,"executed_commands":0,"training_started":False,
        "automatic_retry_performed":False,"protected_file_count":count,
        "interpretation":"Incomplete diagnostic, not task failures; keep any recorded scores without filling missing values"})
    forward.save(RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":count,"changed_paths":[]})
    forward.save(RUN / "planning-update.json",{"explicitly_authorized":True,
        "before":forward.read(RUN / "planning-before.json"),
        "after":{name:forward.digest(ROOT / name) for name in forward.inventory.DOCS},
        "reason":"Record separately requested float32 attempt with stricter startup RAM protection and actual outcome"})


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--run"]:
        configure()
        forward.run()
    elif sys.argv[1:] == ["--record-status"]:
        record_status()
    elif len(sys.argv) == 3 and sys.argv[1] == "--worker" and sys.argv[2] in forward.VARIANTS:
        configure()
        forward.worker(sys.argv[2])
    else:
        raise ValueError("Use --prepare-only, --run, --record-status or internal --worker VARIANT")
