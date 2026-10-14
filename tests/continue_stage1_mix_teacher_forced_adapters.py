"""Score only 011/012 in separate invocations, reusing a verified gold-context base."""

from pathlib import Path
import sys
import time

import run_stage1_mix_teacher_forced_diagnostic as p

ROOT = p.ROOT
RUN = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-adapters-011-012-v1"
BASE = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1-retry2-20261013"
DEPENDENCY = Path(p.__file__)
DEPENDENCY_SHA256 = "b1f643a3d511f18f0c2ad355d5d68371bffd381b98343845e86f260f68d207d4"
ORIGINAL_PINS = dict(p.PINS)
ADAPTERS = {key:value for key,value in p.VARIANTS.items() if key != "base"}
BASE_PINS = {
    BASE / "base/summary.json":"5dc018d32c7e47ad778ebe948ee832a9bb8eaf28545c27b56476494b042d8ea0",
    BASE / "diagnostic-plan.json":"619df7b58c3f8c0720008daee790d907609ec0482c5cd1d89b5a9e5a50efc7f9",
    BASE / "base/model-load.json":"ca9dc0f2d316bba59ecb69a8dbd7f5ef9154849b14ec7d365bc6b1b6f5ecb344",
    BASE / "base/supervisor-result.json":"c7118131c76d687171976230ea9bca70d545ed3b079e01718a8fd0444b73512f",
}


def verify_base():
    for path,value in BASE_PINS.items():
        if p.digest(path) != value:
            raise ValueError("Reused base identity changed")
    plan = p.read(BASE / "diagnostic-plan.json")
    if (plan["dtype"],plan["device"],plan["torch_threads"],plan["revision"]) != ("float32","cpu",4,p.REVISION):
        raise ValueError("Base scoring protocol mismatch")
    if tuple(plan["categories"]) != p.CATEGORIES:
        raise ValueError("Base category protocol mismatch")
    end = p.read(BASE / "base/supervisor-result.json")
    if end["exit_code"] != 0 or end["resource_abort"] is not None or end["restarted"]:
        raise ValueError("Base did not finish cleanly")
    loaded = p.read(BASE / "base/model-load.json")
    if loaded["active_adapters"] or loaded["floating_dtypes"] != ["torch.float32"] or loaded["devices"] != ["cpu"]:
        raise ValueError("Not the raw float32 CPU base")
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    tokens,hashes = [],dict(BASE_PINS)
    for example in examples:
        path = BASE / "base/cases" / (example["source_trajectory_id"] + ".json")
        report = p.read(path)
        if report["trajectory_id"] != example["source_trajectory_id"] or report["split"] != "train":
            raise ValueError("Base trajectory identity mismatch")
        rows = report["tokens"]
        if [row["position"] for row in rows] != p.prediction_positions(example["labels"]):
            raise ValueError("Base causal targets mismatch")
        if any(row["target_id"] != example["labels"][row["position"]] for row in rows):
            raise ValueError("Base target labels mismatch")
        if p.aggregate(rows) != report["metrics"]:
            raise ValueError("Base case metrics do not match token records")
        tokens.extend(rows)
        hashes[path] = p.digest(path)
    summary = p.read(BASE / "base/summary.json")
    if len(examples) != 14 or len(tokens) != 3093 or summary["metrics"] != p.aggregate(tokens):
        raise ValueError("Base summary does not match all fourteen cases")
    return hashes


def configure():
    if p.digest(DEPENDENCY) != DEPENDENCY_SHA256:
        raise ValueError("Frozen forward implementation changed")
    p.RUN = RUN
    p.__file__ = __file__
    p.VARIANTS = dict(ADAPTERS)
    p.safety.START_LIMITS = {**p.safety.START_LIMITS,"physical_free_bytes":14 * 1024**3}
    p.PINS = {**ORIGINAL_PINS,**BASE_PINS,DEPENDENCY:DEPENDENCY_SHA256}
    if (RUN / "diagnostic-plan.json").exists():
        prepared = p.read(RUN / "prepared.json")
        if p.digest(RUN / "diagnostic-plan.json") != prepared["plan_sha256"]:
            raise ValueError("Frozen adapter plan changed")
        if p.digest(RUN / "base-reuse.json") != prepared["base_reuse_sha256"]:
            raise ValueError("Base reuse manifest changed")
        plan = p.read(RUN / "diagnostic-plan.json")
        p.PINS = {ROOT / name:value for name,value in plan["input_pins"].items()}
        for path,value in {**ORIGINAL_PINS,**BASE_PINS,DEPENDENCY:DEPENDENCY_SHA256}.items():
            if p.PINS.get(path) != value:
                raise ValueError("Required input identity pin missing")


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing overwrite or resume")
    configure()
    base_hashes = verify_base()
    p.PINS.update(base_hashes)
    p.prepare()
    p.save(RUN / "base-reuse.json",{
        "source_run":BASE.relative_to(ROOT).as_posix(),"model_reexecuted":False,
        "case_count":14,"supervised_token_count":3093,
        "hashes":{path.relative_to(ROOT).as_posix():value for path,value in base_hashes.items()}})
    p.save(RUN / "prepared.json",{"plan_sha256":p.digest(RUN / "diagnostic-plan.json"),
        "base_reuse_sha256":p.digest(RUN / "base-reuse.json"),
        "new_model_variants":["011","012"],"base_reexecuted":False,
        "resource_wait_seconds_per_variant":180,"automatic_model_retry":False,
        "authorization":"User explicitly requested only 011/012 and reuse of saved raw-base scores"})


def evaluate(variant):
    configure()
    if variant not in ADAPTERS:
        raise ValueError("Only 011/012 are authorized; base must not run")
    if (RUN / variant).exists() or (RUN / (variant+"-startup-blocked.json")).exists():
        raise FileExistsError("No automatic model retry or overwrite")
    p.verify_inputs()
    p.verify_preservation()
    deadline = time.monotonic()+180
    while True:
        snapshot = p.safety.resources()
        low = p.safety.below_limits(snapshot,p.safety.START_LIMITS)
        p.inventory.append(RUN / (variant+"-preload-resources.jsonl"),snapshot)
        if not low:
            break
        if time.monotonic() >= deadline:
            p.save(RUN / (variant+"-startup-blocked.json"),{"resources":snapshot,"limits":p.safety.START_LIMITS})
            raise RuntimeError(f"Startup headroom still unavailable for {variant}: {snapshot}")
        print(f"Waiting for host reclamation before {variant}: {low}",flush=True)
        time.sleep(10)
    p.run_variant(variant)


def finish_scores():
    configure()
    p.verify_inputs()
    verify_base()
    summaries = {"base":p.read(BASE / "base/summary.json")}
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    for variant in ADAPTERS:
        end = p.read(RUN / variant / "supervisor-result.json")
        if end["exit_code"] or end["resource_abort"] is not None or end["restarted"]:
            raise ValueError("Adapter scoring incomplete")
        summaries[variant] = p.read(RUN / variant / "summary.json")
        rows = []
        for example in examples:
            name = example["source_trajectory_id"]+".json"
            base = p.read(BASE / "base/cases" / name)
            report = p.read(RUN / variant / "cases" / name)
            def targets(record):
                return [(t["position"],t["target_id"],t["categories"]) for t in record["tokens"]]
            if targets(report) != targets(base):
                raise ValueError("Scored targets differ across models")
            rows.extend(report["tokens"])
        if len(rows) != 3093 or summaries[variant]["metrics"] != p.aggregate(rows):
            raise ValueError("Adapter summary differs from raw scores")
    p.save(RUN / "comparison.json",{"variants":summaries,"base_reexecuted":False,
        "base_source":BASE.relative_to(ROOT).as_posix(),
        "interpretation":"Gold-history next-token scores on seen data; not autonomous task success or unseen validation"})
    p.save(RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":p.verify_preservation(),"changed_paths":[]})


def seal_status():
    configure()
    count = p.verify_preservation()
    if (RUN / "comparison.json").exists():
        p.save(RUN / "completion.json",{"new_variants":["011","012"],"reused_variant":"base",
            "case_count_per_variant":14,"comparison_variant_count":3,"base_reexecuted":False,
            "protected_file_count":count,"training_started":False,"generation_calls":0,"executed_commands":0})
    else:
        counts,events = {},{}
        for variant in ADAPTERS:
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
            "interpretation":"Partial scores are retained; missing adapter scores are not zeros"})
        p.save(RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":count,"changed_paths":[]})
    p.save(RUN / "planning-update.json",{"explicitly_authorized":True,
        "before":p.read(RUN / "planning-before.json"),"after":{name:p.digest(ROOT / name) for name in p.inventory.DOCS},
        "reason":"Authorized adapter-only continuation with hash-pinned reuse of saved base, unchanged guards"})


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--prepare-only"]:
        prepare()
    elif len(args) == 2 and args[0] in {"--evaluate","--worker"} and args[1] in ADAPTERS:
        if args[0] == "--evaluate":
            evaluate(args[1])
        else:
            configure()
            p.worker(args[1])
    elif args == ["--finish-scores"]:
        finish_scores()
    elif args == ["--seal-status"]:
        seal_status()
    else:
        raise ValueError("Use --prepare-only, --evaluate 011/012, --finish-scores or --seal-status")
