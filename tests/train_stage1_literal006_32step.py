"""Fresh-base literal006-only LoRA; 32 steps, no evaluation or continuation."""

import gc
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256

import export_stage1_literal_edit006_sft as source
import train_stage1_edit005_32step as recording
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan
from capability_capsule.eval.training_provenance import TrainingRunManifest, build_training_dataset_stats
from capability_capsule.training.lora import LoraTrainingConfig, reload_lora_adapter, run_lora_training
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples
from capability_capsule.training.transformers_peft import TransformersPeftAdapterLoader

ROOT = source.ROOT
RUN_ID = "stage1-codex-qwen35-2b-009-literal006-32step"
RUN = ROOT / "runs/training" / RUN_ID
SFT = source.OUTPUT
PUBLICATION = source.publication.OUTPUT
SFT_MANIFEST_DIGEST = "278e4991a0a3d22f79195890c543e21ea0218b7ee10e642a37cd61547ca2d8ea"


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def protected_hashes():
    hashes = source.protected_hashes()
    for path in SFT.iterdir():
        if path.is_file():
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    for path in (source.EVIDENCE, source.publication.EVIDENCE):
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path: digest for path, digest in hashes.items() if not path.startswith(prefix)}


def prepare():
    if sha256((SFT / "manifest.json").read_bytes()).hexdigest() != SFT_MANIFEST_DIGEST:
        raise ValueError("SFT manifest differs from reviewed export")
    publication = load_teacher_dataset_publication(PUBLICATION)
    sft = SFTExportManifest.model_validate_json((SFT / "manifest.json").read_bytes())
    if sft.source_dataset_digest != source.SOURCE_DIGEST or sft.schema_version != "0.3":
        raise ValueError("SFT source/schema mismatch")
    if sft.chat_contract_sha256 != source.CONTRACT_DIGEST or sft.chat_contract != source.contract():
        raise ValueError("SFT v2 contract changed")
    for artifact in sft.artifacts:
        payload = (SFT / artifact.filename).read_bytes()
        if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("SFT integrity mismatch")
    examples = load_sft_examples(SFT / "train.jsonl")
    if len(examples) != 8 or load_sft_examples(SFT / "validation.jsonl"):
        raise ValueError("Expected eight train and zero training validation examples")
    for example, trajectory in zip(examples, publication.dataset.train, strict=True):
        if example.source_trajectory_id != trajectory.trajectory_id or example.source_revision != trajectory.source_revision:
            raise ValueError("SFT source identity mismatch")
    plan = load_teacher_collection_plan(source.publication.collection.PLAN_DIR).plan
    verify_student_target(plan.student_target, artifact_root=ROOT)
    prior = TrainingRunManifest.model_validate_json((recording.RUN / "training-run.json").read_bytes())
    options = {key: prior.hyperparameters[key] for key in (
        "epochs", "max_steps", "learning_rate", "train_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout", "target_modules")}
    options.update(epochs=4, max_steps=32)
    model, revision = sft.tokenizer_id.split("@", 1)
    if (model, revision) != (source.MODEL_ID, source.REVISION):
        raise ValueError("Pinned model revision changed")
    config = LoraTrainingConfig(base_model_id=model, base_model_revision=revision,
        trainer_id=prior.trainer_id, seed=prior.random_seed, **options)
    counts = {e.source_trajectory_id: len(e.input_ids) for e in examples}
    stats = build_training_dataset_stats(publication.dataset, tasks=[a.task for a in plan.assignments],
        dataset_id=sft.source_dataset_id, dataset_digest=sft.source_dataset_digest,
        tokenizer_id=sft.tokenizer_id, token_counter=lambda tr: counts[tr.trajectory_id])
    if stats.train.exact_token_count != 5506 or sum(sum(label != -100 for label in e.labels) for e in examples) != 1588:
        raise ValueError("Reviewed token counts changed")
    parameters = {**options, "sft_export_id": sft.export_id, "sequence_length": 4096,
        "evaluation_strategy": "none; independent checkpoint evaluation",
        "chat_contract_id": sft.chat_contract.contract_id, "chat_contract_sha256": sft.chat_contract_sha256,
        "data_composition": "literal-edit-006-only; fresh base model; no old adapter continuation",
        "expected_effective_epochs": 4.0, "trainable_token_count": 1588,
        "scheduler_policy": "Existing Trainer default linear decay over 32 steps; no warmup",
        "comparison_policy": "New data and exposure per example differ from edit005; not a single-variable control"}
    manifest = TrainingRunManifest(run_id=RUN_ID, output_capsule_id=f"capsule-{RUN_ID}",
        dataset_stats=stats, base_model_id=model, trainer_id=config.trainer_id,
        hardware_id=prior.hardware_id, random_seed=config.seed, started_at=datetime.now(UTC), hyperparameters=parameters)
    if manifest.hardware_id != plan.student_target.hardware.profile_id:
        raise ValueError("Pinned hardware identity changed")
    return config, manifest


def verify_preservation():
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    after = protected_hashes()
    # Historical protection locks recorded files; later authorized runs may add new artifacts.
    changed = sorted(p for p, pinned in before.items() if after.get(p) != pinned)
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
    before = protected_hashes()
    config, manifest = prepare()
    print(f"Starting {RUN_ID}: fresh base, eight examples, 32 steps / four epochs; no validation", flush=True)
    result = run_lora_training(config, training_run=manifest, train_path=SFT / "train.jsonl",
        validation_path=SFT / "validation.jsonl", output_root=RUN.parent, backend=recording.RecordingBackend())
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    if state["global_step"] != 32 or result.checkpoint.step != 32 or abs(state["epoch"] - 4.0) > 1e-8:
        raise ValueError("Training did not complete 32 steps / four measured epochs")
    phase_path = RUN / "training-phase-metrics.json"
    phase = json.loads(phase_path.read_text("utf-8"))
    phase.update(configured_epochs=config.epochs, epoch=state["epoch"])
    phase_path.write_text(json.dumps(phase, indent=2) + "\n", encoding="utf-8")
    save(RUN / "protected-input-hashes.json", before)
    count = verify_preservation()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count, "changed_paths": []})
    print(json.dumps(phase, indent=2), flush=True)
    print(f"Training finished; {count} existing input files byte-unchanged", flush=True)


if __name__ == "__main__":
    main()
