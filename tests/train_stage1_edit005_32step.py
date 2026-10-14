"""Run and independently reload the authorized edit-005 32-step experiment."""

import contextlib
import gc
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.training_provenance import (
    TrainingRunManifest,
    build_training_dataset_stats,
)
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan
from capability_capsule.training.lora import (
    LoraTrainingConfig,
    reload_lora_adapter,
    run_lora_training,
)
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples
import capability_capsule.training.transformers_peft as backend_module


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage1-codex-qwen35-2b-007-edit005-32step"
RUN = ROOT / "runs/training" / RUN_ID
SFT = ROOT / "artifacts/sft/stage1-codex-powershell-edit-005-qwen35-2b-v1"
PUBLICATION = ROOT / "datasets/teacher/published/stage1-codex-powershell-edit-005"


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.terminal.write(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


class RecordingBackend(backend_module.TransformersPeftBackend):
    def train(self, **kwargs):
        run_dir = kwargs["run_dir"]
        original_loader = backend_module._load_training_libraries
        libraries = original_loader()
        backend_module._disable_shadowed_dataset_namespace(libraries.Trainer)
        from transformers import TrainerCallback

        class LogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **extra):
                with (run_dir / "optimizer-log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **(logs or {})}) + "\n")

        class RecordingTrainer(libraries.Trainer):
            def __init__(self, *args, **options):
                super().__init__(*args, **options)
                self.add_callback(LogCallback())

            def train(self, *args, **options):
                result = super().train(*args, **options)
                self.state.save_to_json(str(run_dir / "trainer-state.json"))
                return result

        RecordingTrainer.__module__ = libraries.Trainer.__module__
        backend_module._load_training_libraries = lambda: libraries._replace(Trainer=RecordingTrainer)
        try:
            with (run_dir / "training-console.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    return super().train(**kwargs)
        finally:
            backend_module._load_training_libraries = original_loader


def reload_only() -> None:
    model = reload_lora_adapter(
        RUN / "checkpoint.json", loader=backend_module.TransformersPeftAdapterLoader(),
    )
    report = {
        "schema_version": "0.1", "run_id": RUN_ID,
        "base_model_revision_verified": True,
        "adapter_hashes_verified": True,
        "loader_class": type(model).__name__,
        "active_adapters": list(model.active_adapters),
        "independent_process": True,
    }
    with (RUN / "adapter-reload-validation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    del model
    gc.collect()


def main() -> None:
    if sys.argv[1:] == ["--reload-only"]:
        reload_only()
        return
    if sys.argv[1:]:
        raise ValueError("Only --reload-only is supported")
    if RUN.exists():
        raise FileExistsError(RUN)
    verified = load_teacher_dataset_publication(PUBLICATION)
    sft_manifest = SFTExportManifest.model_validate_json((SFT / "manifest.json").read_bytes())
    if sft_manifest.source_dataset_digest != sha256((PUBLICATION / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("SFT source digest mismatch")
    if sft_manifest.schema_version != "0.3":
        raise ValueError("This experiment requires normalized schema 0.3")
    for artifact in sft_manifest.artifacts:
        payload = (SFT / artifact.filename).read_bytes()
        if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("SFT artifact integrity mismatch")
    train = load_sft_examples(SFT / "train.jsonl")
    validation = load_sft_examples(SFT / "validation.jsonl")
    if len(train) != 12 or validation:
        raise ValueError("Expected 12 train and no training-period validation")
    examples_by_id = {item.source_trajectory_id: item for item in train}
    plan = load_teacher_collection_plan(ROOT / "plans/teacher/stage1-codex-powershell-edit-005").plan
    stats = build_training_dataset_stats(
        verified.dataset, tasks=[assignment.task for assignment in plan.assignments],
        dataset_id=verified.manifest.dataset_id, dataset_digest=sft_manifest.source_dataset_digest,
        tokenizer_id=sft_manifest.tokenizer_id,
        token_counter=lambda trajectory: len(examples_by_id[trajectory.trajectory_id].input_ids),
    )
    prior = TrainingRunManifest.model_validate_json(
        (ROOT / "runs/training/stage1-codex-qwen35-2b-006-turns-v2/training-run.json").read_bytes()
    )
    model_id, revision = sft_manifest.tokenizer_id.split("@", 1)
    config = LoraTrainingConfig(
        base_model_id=model_id, base_model_revision=revision, trainer_id=prior.trainer_id,
        seed=prior.random_seed,
        **{key: prior.hyperparameters[key] for key in (
            "epochs", "max_steps", "learning_rate", "train_batch_size", "eval_batch_size",
            "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout", "target_modules",
        )},
    )
    hyperparameters = dict(prior.hyperparameters)
    hyperparameters.update({
        "sft_export_id": sft_manifest.export_id,
        "comparison_policy": "32 optimizer steps fixed against run 006",
        "data_composition": "edit-005-only; fresh base model",
        "expected_effective_epochs": 32 / 12,
    })
    manifest = TrainingRunManifest(
        run_id=RUN_ID, output_capsule_id=f"capsule-{RUN_ID}", dataset_stats=stats,
        base_model_id=model_id, trainer_id=config.trainer_id, hardware_id=prior.hardware_id,
        random_seed=config.seed, started_at=datetime.now(UTC), hyperparameters=hyperparameters,
    )
    print(f"Starting {RUN_ID}: 12 examples, {stats.train.exact_token_count} tokens, 32 steps", flush=True)
    result = run_lora_training(
        config, training_run=manifest,
        train_path=SFT / "train.jsonl", validation_path=SFT / "validation.jsonl",
        output_root=ROOT / "runs/training", backend=RecordingBackend(),
    )
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    if result.checkpoint.step != 32 or state["global_step"] != 32:
        raise ValueError("Training did not finish the authorized 32 steps")
    phase_path = RUN / "training-phase-metrics.json"
    phase = json.loads(phase_path.read_text("utf-8"))
    phase["configured_epochs"] = config.epochs
    phase["epoch"] = state["epoch"]
    phase_path.write_text(json.dumps(phase, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(phase, indent=2), flush=True)


if __name__ == "__main__":
    main()
