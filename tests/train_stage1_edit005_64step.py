"""Fresh-base 64-step control; preserve every existing experiment artifact."""

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import train_stage1_edit005_32step as prior_script
import run_stage1_007_evaluation as prior_eval
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan
from capability_capsule.eval.training_provenance import TrainingRunManifest, build_training_dataset_stats
from capability_capsule.training.lora import LoraTrainingConfig, run_lora_training
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage1-codex-qwen35-2b-008-edit005-64step"
RUN = ROOT / "runs/training" / RUN_ID
SFT = prior_script.SFT
PUBLICATION = prior_script.PUBLICATION
BASELINE = prior_script.RUN
BASE_EVALUATION = prior_eval.RUN


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protected_hashes():
    hashes = {}
    directories = [ROOT / "datasets/teacher", ROOT / "artifacts/sft", BASELINE, BASE_EVALUATION]
    for case in prior_eval.load_coding_evaluation_suite(prior_eval.SUITE).cases:
        directories.append(ROOT / case.fixture_root)
    plan = load_teacher_collection_plan(ROOT / "plans/teacher/stage1-codex-powershell-edit-005").plan
    directories.extend(ROOT / a.authorized_fixture_root for a in plan.assignments)
    for directory in directories:
        for path in directory.rglob("*"):
            if path.is_file() and not any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    for path in (prior_eval.SUITE, prior_eval.EXECUTOR):
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    if sys.argv[1:] == ["--reload-only"]:
        old_run, old_id = prior_script.RUN, prior_script.RUN_ID
        prior_script.RUN, prior_script.RUN_ID = RUN, RUN_ID
        try:
            prior_script.reload_only()
        finally:
            prior_script.RUN, prior_script.RUN_ID = old_run, old_id
        return
    if sys.argv[1:]:
        raise ValueError("Only --reload-only is supported")
    if RUN.exists():
        raise FileExistsError(RUN)
    before = protected_hashes()
    publication = load_teacher_dataset_publication(PUBLICATION)
    sft = SFTExportManifest.model_validate_json((SFT / "manifest.json").read_bytes())
    if sft.source_dataset_digest != sha256((PUBLICATION / "manifest.json").read_bytes()).hexdigest():
        raise ValueError("SFT publication mismatch")
    for artifact in sft.artifacts:
        payload = (SFT / artifact.filename).read_bytes()
        if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("SFT integrity mismatch")
    examples = load_sft_examples(SFT / "train.jsonl")
    if len(examples) != 12 or load_sft_examples(SFT / "validation.jsonl"):
        raise ValueError("Expected unchanged 12 train and 0 validation examples")
    old = TrainingRunManifest.model_validate_json((BASELINE / "training-run.json").read_bytes())
    model, revision = sft.tokenizer_id.split("@", 1)
    options = {key: old.hyperparameters[key] for key in (
        "epochs", "max_steps", "learning_rate", "train_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout", "target_modules")}
    options["max_steps"] = 64
    config = LoraTrainingConfig(base_model_id=model, base_model_revision=revision,
                                trainer_id=old.trainer_id, seed=old.random_seed, **options)
    counts = {e.source_trajectory_id: len(e.input_ids) for e in examples}
    plan = load_teacher_collection_plan(ROOT / "plans/teacher/stage1-codex-powershell-edit-005").plan
    stats = build_training_dataset_stats(publication.dataset, tasks=[a.task for a in plan.assignments],
        dataset_id=sft.source_dataset_id, dataset_digest=sft.source_dataset_digest,
        tokenizer_id=sft.tokenizer_id, token_counter=lambda tr: counts[tr.trajectory_id])
    hyperparameters = {**old.hyperparameters, "max_steps": 64, "expected_effective_epochs": 64 / 12,
        "comparison_policy": "Fresh-base 64 vs 32 max_steps on identical edit005-only SFT",
        "scheduler_policy": "Unchanged Trainer default linear decay; horizon grows from 32 to 64 steps, not a continuation.",
        "baseline_training_run_id": old.run_id}
    manifest = TrainingRunManifest(run_id=RUN_ID, output_capsule_id=f"capsule-{RUN_ID}",
        dataset_stats=stats, base_model_id=model, trainer_id=old.trainer_id, hardware_id=old.hardware_id,
        random_seed=old.random_seed, started_at=datetime.now(UTC), hyperparameters=hyperparameters)
    print(f"Starting {RUN_ID}: 12 unchanged examples, 64 steps; no training validation", flush=True)
    result = run_lora_training(config, training_run=manifest, train_path=SFT / "train.jsonl",
        validation_path=SFT / "validation.jsonl", output_root=ROOT / "runs/training",
        backend=prior_script.RecordingBackend())
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    if state["global_step"] != 64 or result.checkpoint.step != 64:
        raise ValueError("64 optimizer steps were not completed")
    phase = json.loads((RUN / "training-phase-metrics.json").read_text("utf-8"))
    phase.update(configured_epochs=config.epochs, epoch=state["epoch"])
    save(RUN / "training-phase-metrics.json", phase)
    save(RUN / "protected-input-hashes.json", before)
    after = protected_hashes()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    save(RUN / "input-preservation.json", {"unchanged": not changed,
        "protected_file_count": len(before), "changed_paths": changed})
    if changed:
        raise ValueError("Protected inputs changed")
    print(json.dumps(phase, indent=2), flush=True)


if __name__ == "__main__":
    main()
