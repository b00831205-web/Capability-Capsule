"""One authorized memory-waived attempt; retain frozen precision probe and disk gates."""
from pathlib import Path
import sys

import run_stage1_mix012_first_edit_precision as p

ROOT = p.ROOT
ORIGINAL = p.RUN
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-first-edit-precision-memory-override-v1"
DEPENDENCY = Path(p.__file__)
START = {"c_free_bytes":8 * 1024**3}
STOP = {"c_free_bytes":4 * 1024**3}


def configure():
    p.RUN = RUN
    p.START = dict(START)
    p.STOP = dict(STOP)
    p.verify = verify
    p.__file__ = __file__  # Child workers must enter the same scoped configuration.


def verify():
    plan = p.read(RUN/"diagnostic-plan.json")
    for name,expected in plan["input_pins"].items():
        if p.digest(ROOT/name) != expected:
            raise ValueError("Frozen input changed: " + name)
    if (plan["start_limits"] != START or plan["stop_limits"] != STOP
            or not plan["memory_guard_override"] or not plan["disk_guard_retained"]):
        raise ValueError("Scoped guard policy changed")
    original = p.read(ORIGINAL/"diagnostic-plan.json")
    for key in ("cases","contract","variant_order","max_new_tokens","do_sample","use_cache",
                "seed","torch_threads","device","adapter_autocast_dtype"):
        if plan[key] != original[key]:
            raise ValueError("Frozen comparison design changed: " + key)
    authorization = p.read(RUN/"execution-authorization.json")
    if (authorization["authorized_variants"] != list(p.VARIANTS)
            or not authorization["memory_guard_override"] or not authorization["disk_guard_retained"]
            or authorization["command_execution"] or authorization["training"]):
        raise ValueError("Authorization scope changed")
    return plan


def prepare():
    if RUN.exists():
        raise FileExistsError("Never overwrite or retry a previous attempt")
    original = p.verify()
    result = p.read(ORIGINAL/"comparison.json")
    if any(v["recorded_cases"] != 0 or v["supervisor"]["status"] != "startup_blocked"
           for v in result["variants"].values()):
        raise ValueError("Original attempt has model output")
    configure()
    p.prior.RUN = RUN
    protected = p.prior.protected_hashes()
    RUN.mkdir(parents=True)
    p.save(RUN/"protected-input-hashes.json",protected)
    p.save(RUN/"planning-before.json",{name:p.digest(ROOT/name) for name in p.prior.inventory.DOCS})
    p.save(RUN/"execution-authorization.json",{
        "basis":"User explicitly authorized ignoring memory thresholds for this diagnosis, retaining disk protection",
        "authorized_variants":list(p.VARIANTS),"memory_guard_override":True,"disk_guard_retained":True,
        "memory_telemetry_retained":True,"system_configuration_changed":False,
        "disabled_memory_checks":["physical_free_bytes","commit_free_bytes","wsl_available_bytes"],
        "start_limits":START,"stop_limits":STOP,"command_execution":False,"training":False,
        "automatic_retry":False,"scope":"Only this new eight-turn diagnostic attempt; no persistent waiver"})
    pins = dict(original["input_pins"])
    for path in (DEPENDENCY,Path(__file__),ORIGINAL/"diagnostic-plan.json",ORIGINAL/"comparison.json",
                 RUN/"execution-authorization.json"):
        pins[path.relative_to(ROOT).as_posix()] = p.digest(path)
    p.save(RUN/"diagnostic-plan.json",{**original,"input_pins":pins,"start_limits":START,"stop_limits":STOP,
        "memory_guard_override":True,"disk_guard_retained":True,"memory_telemetry_retained":True,
        "system_configuration_changed":False,"original_attempt_preserved":True})
    verify()
    print(f"Prepared memory-waived eight-turn precision probe; {len(protected)} old files protected",flush=True)


def seal():
    verify()
    p.save(RUN/"planning-update.json",{"explicitly_authorized":True,
        "before":p.read(RUN/"planning-before.json"),
        "after":{name:p.digest(ROOT/name) for name in p.prior.inventory.DOCS},
        "reason":"User-authorized memory-waived precision probe; disk protection retained, no system changes"})


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--prepare"]:
        prepare()
    elif len(args) == 2 and args[0] in ("--evaluate","--worker") and args[1] in p.VARIANTS:
        configure()
        (p.evaluate if args[0] == "--evaluate" else p.worker)(args[1])
    elif args == ["--finish"]:
        configure()
        p.finish()
    elif args == ["--seal"]:
        configure()
        seal()
    else:
        raise ValueError("Use --prepare, --evaluate float32/bfloat16, --finish, --seal")
