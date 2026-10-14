"""Fresh-base 64-step exposure control on the exact checkpoint-009 SFT."""

import gc
import json
import sys
from datetime import UTC, datetime

import train_stage1_literal006_32step as baseline
from capability_capsule.eval.training_provenance import TrainingRunManifest
from capability_capsule.training.lora import LoraTrainingConfig, reload_lora_adapter, run_lora_training
from capability_capsule.training.transformers_peft import TransformersPeftAdapterLoader

ROOT = baseline.ROOT
RUN_ID = "stage1-codex-qwen35-2b-010-literal006-64step-retry1"
RUN = ROOT / "runs/training" / RUN_ID
SFT = baseline.SFT
PUBLICATION = baseline.PUBLICATION
BASELINE = baseline.RUN
save = baseline.save


def protected_hashes():
    hashes = baseline.protected_hashes()
    for path in BASELINE.rglob("*"):
        if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            hashes[path.relative_to(ROOT).as_posix()] = baseline.sha256(path.read_bytes()).hexdigest()
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path: digest for path, digest in hashes.items() if not path.startswith(prefix)}


def prepare():
    config32, candidate = baseline.prepare()
    recorded = TrainingRunManifest.model_validate_json((BASELINE / "training-run.json").read_bytes())
    if (candidate.hyperparameters != recorded.hyperparameters or candidate.dataset_stats != recorded.dataset_stats
            or candidate.base_model_id != recorded.base_model_id or candidate.random_seed != recorded.random_seed
            or candidate.hardware_id != recorded.hardware_id or candidate.trainer_id != recorded.trainer_id):
        raise ValueError("Current inputs/config differ from the recorded 32-step baseline")
    config = LoraTrainingConfig.model_validate({**config32.model_dump(), "max_steps": 64})
    parameters = {**recorded.hyperparameters, "max_steps": 64, "expected_effective_epochs": 8.0,
        "scheduler_policy": "Existing Trainer default linear decay; horizon 32 -> 64 steps, no warmup; not a continuation or identical LR prefix",
        "comparison_policy": "Fresh-base 64 vs 32 max_steps, identical literal006-only SFT, seed, model revision and LoRA settings",
        "baseline_training_run_id": recorded.run_id}
    manifest = TrainingRunManifest(run_id=RUN_ID, output_capsule_id=f"capsule-{RUN_ID}",
        dataset_stats=recorded.dataset_stats, base_model_id=recorded.base_model_id,
        trainer_id=recorded.trainer_id, hardware_id=recorded.hardware_id,
        random_seed=recorded.random_seed, started_at=datetime.now(UTC), hyperparameters=parameters)
    return config, manifest


def verify_preservation():
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    current = protected_hashes()
    changed = sorted(path for path, digest in before.items() if current.get(path) != digest)
    if changed:
        raise ValueError(f"Protected inputs changed: {changed}")
    return len(before)


def reload_only():
    if (RUN / "adapter-reload-validation.json").exists():
        raise FileExistsError("Independent reload already recorded")
    verify_preservation()
    model = reload_lora_adapter(RUN / "checkpoint.json", loader=TransformersPeftAdapterLoader())
    report = {"run_id": RUN_ID, "independent_process": True, "base_model_revision_verified": True,
        "adapter_hashes_verified": True, "loader_class": type(model).__name__,
        "active_adapters": list(model.active_adapters), "protected_file_count": verify_preservation()}
    save(RUN / "adapter-reload-validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    del model
    gc.collect()


def main():
    if sys.argv[1:] == ["--reload-only"]:
        reload_only()
        return
    if sys.argv[1:]:
        raise ValueError("Only --reload-only is supported")
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing run")
    baseline.verify_preservation()
    before = protected_hashes()
    config, manifest = prepare()
    print(f"Starting {RUN_ID}: identical eight examples, fresh base, 64 updates / eight effective epochs; no validation", flush=True)
    result = run_lora_training(config, training_run=manifest, train_path=SFT / "train.jsonl",
        validation_path=SFT / "validation.jsonl", output_root=RUN.parent,
        backend=baseline.recording.RecordingBackend())
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    if state["global_step"] != 64 or result.checkpoint.step != 64 or abs(state["epoch"] - 8.0) > 1e-8:
        raise ValueError("Training did not complete 64 updates / eight measured epochs")
    phase_path = RUN / "training-phase-metrics.json"
    phase = json.loads(phase_path.read_text("utf-8"))
    phase.update(configured_epochs=config.epochs, epoch=state["epoch"])
    phase_path.write_text(json.dumps(phase, indent=2) + "\n", encoding="utf-8")
    save(RUN / "protected-input-hashes.json", before)
    count = verify_preservation()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count, "changed_paths": []})
    print(json.dumps(phase, indent=2), flush=True)
    print(f"Training finished; {count} existing files byte-unchanged", flush=True)


if __name__ == "__main__":
    main()
