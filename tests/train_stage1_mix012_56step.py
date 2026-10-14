"""Authorized fresh-base 56-step control; never performs task evaluation."""

import gc
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

import train_stage1_mix011_28step as baseline
import recover_stage1_mix011_train_fit_diagnostic as safety
from capability_capsule.eval.training_provenance import TrainingRunManifest
from capability_capsule.training.lora import LoraTrainingConfig, reload_lora_adapter, run_lora_training
from capability_capsule.training.transformers_peft import TransformersPeftAdapterLoader

ROOT = baseline.ROOT
RUN_ID = "stage1-codex-qwen35-2b-012-mix006007-56step"
RUN = ROOT / "runs/training" / RUN_ID
SFT = baseline.SFT
save = baseline.save
digest = baseline.digest
DOCS = ("MVP_ROADMAP.md", "PENDING_CONFIGURATION_OPTIMIZATIONS.md")
IGNORE_MEMORY_GUARD = False


def resource_limits(start):
    limits = safety.START_LIMITS if start else safety.STOP_LIMITS
    return {key: value for key, value in limits.items() if not IGNORE_MEMORY_GUARD or key == "c_free_bytes"}


def prepare():
    original, candidate = baseline.prepare()
    recorded = TrainingRunManifest.model_validate_json((baseline.RUN / "training-run.json").read_bytes())
    if (candidate.hyperparameters != recorded.hyperparameters or candidate.dataset_stats != recorded.dataset_stats
            or candidate.base_model_id != recorded.base_model_id or candidate.trainer_id != recorded.trainer_id
            or candidate.hardware_id != recorded.hardware_id or candidate.random_seed != recorded.random_seed):
        raise ValueError("Inputs no longer match recorded 011")
    if digest(baseline.RUN / "checkpoint.json") != "30b10ef7364c6b1ea7cb94a72d2d064b84318a5969d078a2b4e0598ed44bba52":
        raise ValueError("Baseline checkpoint changed")
    config = LoraTrainingConfig.model_validate({**original.model_dump(), "max_steps": 56, "epochs": 4})
    parameters = {**recorded.hyperparameters, "max_steps": 56, "epochs": 4,
        "expected_effective_epochs": 4.0, "supervised_token_exposures": 3093 * 4,
        "baseline_training_run_id": recorded.run_id,
        "scheduler_policy": "Existing Trainer default linear decay, 56 steps, no warmup; not identical LR prefix to 28-step run",
        "comparison_policy": "Fresh-base same-SFT budget comparison against 011; schedule horizon changes too; not an isolated causal test",
        "checkpoint_selection": "Final authorized step 56 only; no validation-based selection",
        "model_evaluation_authorized": False,
        "memory_guard_override": IGNORE_MEMORY_GUARD,
        "authorization_scope": "New 012 training directory, relevant tests and standing planning updates only; no task evaluation"}
    manifest = TrainingRunManifest(run_id=RUN_ID, output_capsule_id="capsule-" + RUN_ID,
        dataset_stats=recorded.dataset_stats, base_model_id=recorded.base_model_id,
        trainer_id=recorded.trainer_id, hardware_id=recorded.hardware_id,
        random_seed=recorded.random_seed, started_at=datetime.now(UTC), hyperparameters=parameters)
    return config, manifest


def protected_hashes():
    # Includes the old interrupted diagnostic and its completed recovery, not only 011 training.
    previous = safety.diagnostic.RUN
    safety.diagnostic.RUN = RUN
    try:
        hashes = safety.diagnostic.protected_hashes()
    finally:
        safety.diagnostic.RUN = previous
    hashes.update({p: digest(ROOT / p) for p in baseline.protected_hashes() if p not in DOCS
                   and not p.startswith(RUN.relative_to(ROOT).as_posix() + "/")})
    return hashes


def verify_preservation():
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    changed = [p for p, expected in before.items() if not (ROOT / p).is_file() or digest(ROOT / p) != expected]
    if changed:
        raise ValueError(f"Protected inputs changed: {changed}")
    return len(before)


class RecordingBackend(baseline.recording.RecordingBackend):
    def __init__(self, before):
        self.before = before

    def train(self, **kwargs):
        save(RUN / "protected-input-hashes.json", self.before)
        save(RUN / "execution-authorization.json", {"training_authorized": True,
            "fresh_base": True, "max_steps": 56, "epochs": 4,
            "model_evaluation_authorized": False, "automatic_extension_authorized": False,
            "memory_guard_override": IGNORE_MEMORY_GUARD,
            "override_basis": "User explicitly requested ignoring memory protection for this training" if IGNORE_MEMORY_GUARD else None,
            "disk_guard_retained": True, "memory_telemetry_retained": True})
        snapshot = safety.resources()
        save(RUN / "preload-resources.json", snapshot)
        if safety.below_limits(snapshot, resource_limits(True)):
            raise RuntimeError("Insufficient resources immediately before training model load")
        return super().train(**kwargs)


def train_worker():
    if RUN.exists():
        raise FileExistsError("Refusing overwrite or resume")
    config, manifest = prepare()
    before = protected_hashes()
    print(f"Starting {RUN_ID}: 14 records, fresh base, 56 updates / four epochs; no validation", flush=True)
    result = run_lora_training(config, training_run=manifest, train_path=SFT / "train.jsonl",
        validation_path=SFT / "validation.jsonl", output_root=RUN.parent, backend=RecordingBackend(before))
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    logs = [json.loads(line) for line in (RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    losses = [entry for entry in logs if "loss" in entry]
    if (result.checkpoint.step != 56 or state["global_step"] != 56 or abs(state["epoch"] - 4) > 1e-8
            or [entry["step"] for entry in losses] != list(range(1, 57))
            or any("eval_loss" in entry for entry in logs)):
        raise ValueError("Training did not complete the authorized budget without validation")
    count = verify_preservation()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count, "changed_paths": []})
    phase = json.loads((RUN / "training-phase-metrics.json").read_text("utf-8"))
    summary = {**phase, "run_id": RUN_ID, "epoch": state["epoch"], "configured_epochs": 4,
        "last_logged_loss": losses[-1]["loss"], "protected_file_count": count,
        "checkpoint_sha256": digest(RUN / "checkpoint.json"), "model_evaluation_started": False}
    save(RUN / "training-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def reload_worker():
    if (RUN / "adapter-reload-validation.json").exists():
        raise FileExistsError("Independent reload already recorded")
    verify_preservation()
    model = reload_lora_adapter(RUN / "checkpoint.json", loader=TransformersPeftAdapterLoader())
    save(RUN / "adapter-reload-validation.json", {"run_id": RUN_ID, "independent_process": True,
        "base_model_revision_verified": True, "adapter_hashes_verified": True,
        "active_adapters": list(model.active_adapters), "generated_tokens": 0,
        "protected_file_count": verify_preservation()})
    del model
    gc.collect()


def supervise(mode):
    marker = RUN / (mode + "-supervisor-result.json")
    if marker.exists() or (mode == "train" and RUN.exists()) or (mode == "reload" and (RUN / "adapter-reload-validation.json").exists()):
        raise FileExistsError("Refusing repeated execution")
    snapshot = safety.resources()
    if safety.below_limits(snapshot, resource_limits(True)):
        raise RuntimeError(f"Training/reload resource gate not met: {snapshot}")
    attempt = {"started_at": safety.timestamp(), "resources": snapshot, "mode": mode,
        "memory_guard_override": IGNORE_MEMORY_GUARD, "start_limits": resource_limits(True),
        "stop_limits": resource_limits(False)}
    worker_command = [sys.executable, __file__, "--" + mode + "-worker"]
    if IGNORE_MEMORY_GUARD:
        worker_command.append("--ignore-memory-guard")
    process = subprocess.Popen(worker_command, start_new_session=True)
    samples, stopped = [], None
    while process.poll() is None:
        try:
            sample = safety.resources()
            samples.append(sample)
            low = safety.below_limits(sample, resource_limits(False))
            if low:
                stopped = {"reason": "Resource threshold crossed", "fields": low, "snapshot": sample}
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
    code = process.wait()
    RUN.mkdir(parents=True, exist_ok=True)
    save(RUN / (mode + "-supervisor-attempt.json"), attempt)
    save(RUN / (mode + "-resource-samples.json"), samples)
    save(marker, {"exit_code": code, "resource_abort": stopped, "model_restarted": False,
        "completed_at": safety.timestamp()})
    if code or stopped:
        raise RuntimeError(f"Owned worker stopped: {code}, {stopped}")


def main():
    global IGNORE_MEMORY_GUARD
    args = sys.argv[1:]
    if "--ignore-memory-guard" in args:
        IGNORE_MEMORY_GUARD = True
        args = [arg for arg in args if arg != "--ignore-memory-guard"]
    if args == ["--check"]:
        config, manifest = prepare()
        print(config.model_dump_json(indent=2), flush=True)
        print(manifest.model_dump_json(indent=2), flush=True)
    elif args == ["--train-worker"]:
        train_worker()
    elif args == ["--reload-worker"]:
        reload_worker()
    elif args == ["--reload-only"]:
        supervise("reload")
    elif not args:
        supervise("train")
    else:
        raise ValueError("Use --check or --reload-only, optionally with --ignore-memory-guard")


if __name__ == "__main__":
    main()
