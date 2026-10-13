"""Explicit one-task memory waiver; retain disk protection and original float32 scoring."""

from pathlib import Path
import sys

import continue_stage1_mix_teacher_forced_adapters as previous

p = previous.p
ROOT = p.ROOT
PRIOR = previous.RUN
RUN = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-adapters-011-012-memory-override-v1"
DEPENDENCY = Path(previous.__file__)
DEPENDENCY_SHA256 = "6083cd1694dc1ac1f6ec32bedb04b188d009e4d444ce95d9cee52ee744ad4b97"
PREVIOUS_CONFIGURE = previous.configure
START_LIMITS = {"c_free_bytes":8 * 1024**3}
STOP_LIMITS = {"c_free_bytes":4 * 1024**3}


def configure():
    if p.digest(DEPENDENCY) != DEPENDENCY_SHA256:
        raise ValueError("Frozen adapter-only implementation changed")
    previous.RUN = RUN
    previous.__file__ = __file__
    previous.configure = configure
    PREVIOUS_CONFIGURE()
    p.safety.START_LIMITS = dict(START_LIMITS)
    p.safety.STOP_LIMITS = dict(STOP_LIMITS)
    if (RUN / "diagnostic-plan.json").exists():
        plan = p.read(RUN / "diagnostic-plan.json")
        prepared = p.read(RUN / "prepared.json")
        if p.PINS.get(DEPENDENCY) != DEPENDENCY_SHA256:
            raise ValueError("Adapter dependency pin missing")
        if p.digest(RUN / "execution-authorization.json") != prepared["authorization_sha256"]:
            raise ValueError("Scoped memory authorization changed")
        auth = p.read(RUN / "execution-authorization.json")
        if (plan["start_limits"] != START_LIMITS or plan["stop_limits"] != STOP_LIMITS
                or not plan["memory_guard_override"] or not auth["memory_guard_override"]
                or auth["authorized_variants"] != ["011","012"] or not auth["disk_guard_retained"]):
            raise ValueError("Memory waiver scope or retained disk limits differ")
    else:
        p.PINS[DEPENDENCY] = DEPENDENCY_SHA256


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing overwrite/resume")
    configure()
    base_hashes = previous.verify_base()
    p.PINS.update(base_hashes)
    p.verify_inputs()
    protected = p.protected_hashes()
    RUN.mkdir(parents=True)
    p.save(RUN / "protected-input-hashes.json",protected)
    p.save(RUN / "planning-before.json",{name:p.digest(ROOT / name) for name in p.inventory.DOCS})
    old_plan = p.read(PRIOR / "diagnostic-plan.json")
    plan = {**old_plan,"input_pins":{path.relative_to(ROOT).as_posix():value for path,value in p.PINS.items()},
        "runner_sha256":p.digest(Path(__file__)),"start_limits":START_LIMITS,"stop_limits":STOP_LIMITS,
        "memory_guard_override":True,"disk_guard_retained":True,
        "scope":"Explicitly authorized memory-waived adapter-only forward diagnostic; base reused, never generated commands or training",
        "memory_telemetry_retained":True,"system_configuration_changed":False}
    p.save(RUN / "diagnostic-plan.json",plan)
    p.save(RUN / "base-reuse.json",{
        "source_run":previous.BASE.relative_to(ROOT).as_posix(),"model_reexecuted":False,
        "case_count":14,"supervised_token_count":3093,
        "hashes":{path.relative_to(ROOT).as_posix():value for path,value in base_hashes.items()}})
    p.save(RUN / "execution-authorization.json",{
        "basis":"User explicitly requested: ignore memory thresholds and continue",
        "authorized_variants":["011","012"],"memory_guard_override":True,
        "disabled_memory_checks":["physical_free_bytes","commit_free_bytes","wsl_available_bytes"],
        "disk_guard_retained":True,"start_limits":START_LIMITS,"stop_limits":STOP_LIMITS,
        "memory_telemetry_retained":True,"system_configuration_changed":False,
        "base_reexecuted":False,"training_authorized":False,"generation_authorized":False,
        "command_execution_authorized":False,"automatic_model_retry_authorized":False,
        "authorization_scope":"This new diagnostic only; no persistent/system-wide memory override"})
    p.save(RUN / "prepared.json",{"plan_sha256":p.digest(RUN / "diagnostic-plan.json"),
        "base_reuse_sha256":p.digest(RUN / "base-reuse.json"),
        "authorization_sha256":p.digest(RUN / "execution-authorization.json"),
        "new_model_variants":["011","012"],"base_reexecuted":False,
        "resource_wait_seconds_per_variant":180,"automatic_model_retry":False})
    print(f"Prepared scoped memory waiver, disk gates retained; {len(protected)} old files protected",flush=True)


def seal_status():
    configure()
    count = p.verify_preservation()
    if (RUN / "comparison.json").exists():
        p.save(RUN / "completion.json",{"new_variants":["011","012"],"reused_variant":"base",
            "case_count_per_variant":14,"comparison_variant_count":3,"base_reexecuted":False,
            "protected_file_count":count,"training_started":False,"generation_calls":0,"executed_commands":0,
            "memory_guard_override":True,"disk_guard_retained":True,"system_configuration_changed":False})
    else:
        counts,events = {},{}
        for variant in previous.ADAPTERS:
            directory = RUN / variant
            end = directory / "supervisor-result.json"
            if directory.exists() and not end.exists():
                raise ValueError("Adapter process exit not confirmed")
            counts[variant] = len(list((directory / "cases").glob("*.json")))
            if end.exists():
                events[variant] = p.read(end)
            blocked = RUN / (variant+"-startup-blocked.json")
            if blocked.exists():
                events[variant] = {"startup_blocked":p.read(blocked)}
        p.save(RUN / "interruption.json",{"recorded_adapter_case_counts":counts,"variant_events":events,
            "reused_base_case_count":14,"base_reexecuted":False,"diagnostic_comparison_completed":False,
            "memory_guard_override":True,"disk_guard_retained":True,
            "interpretation":"Keep partial scores; missing scores are not zero task scores"})
    if not (RUN / "input-preservation.json").exists():
        p.save(RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":count,"changed_paths":[]})
    p.save(RUN / "planning-update.json",{"explicitly_authorized":True,
        "before":p.read(RUN / "planning-before.json"),"after":{name:p.digest(ROOT / name) for name in p.inventory.DOCS},
        "reason":"User-authorized memory waiver for adapter-only forward diagnosis, disk protection and baseline preserved"})


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--prepare-only"]:
        prepare()
    elif len(args) == 2 and args[0] in {"--evaluate","--worker"} and args[1] in previous.ADAPTERS:
        configure()
        if args[0] == "--evaluate":
            previous.evaluate(args[1])
        else:
            p.worker(args[1])
    elif args == ["--finish-scores"]:
        configure()
        previous.finish_scores()
    elif args == ["--seal-status"]:
        seal_status()
    else:
        raise ValueError("Use --prepare-only, --evaluate 011/012, --finish-scores or --seal-status")
