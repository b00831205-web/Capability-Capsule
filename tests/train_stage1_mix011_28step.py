"""Execute the separately authorized frozen 28-update pilot; no validation inference."""

import gc
import json
import sys
from datetime import UTC, datetime

import freeze_stage1_mix011_validation as frozen
import train_stage1_edit005_32step as recording
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan
from capability_capsule.eval.training_provenance import TrainingRunManifest, build_training_dataset_stats
from capability_capsule.training.lora import LoraTrainingConfig, reload_lora_adapter, run_lora_training
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples
from capability_capsule.training.transformers_peft import TransformersPeftAdapterLoader

ROOT = frozen.ROOT
RUN_ID = "stage1-codex-qwen35-2b-011-mix006007-28step"
RUN = ROOT / "runs/training" / RUN_ID
SFT = frozen.mix.OUTPUT
PUBLICATION = frozen.mix.PUBLICATION
FREEZE_DIGEST = "74683cd9ab1a9623c08e7cd124d27946f84a091844080579c37e570a1471742e"
digest = frozen.digest
save = frozen.save


def protected_hashes():
    hashes = frozen.protected_hashes()
    for path in frozen.OUTPUT.rglob("*"):
        if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    for name in ("MVP_ROADMAP.md", "PENDING_CONFIGURATION_OPTIMIZATIONS.md"):
        hashes[name] = digest(ROOT / name)
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path:value for path,value in hashes.items() if not path.startswith(prefix)}


def prepare():
    if digest(frozen.OUTPUT / "freeze-manifest.json") != FREEZE_DIGEST:
        raise ValueError("Frozen validation manifest changed")
    frozen.verify_freeze()
    policy = json.loads((frozen.OUTPUT / "experiment-plan.json").read_text("utf-8"))
    if policy["experiment_id"] != RUN_ID or digest(SFT / "manifest.json") != policy["sft_manifest_sha256"]:
        raise ValueError("Frozen experiment/SFT mismatch")
    for path, key in ((frozen.FIXED, "fixed_suite_sha256"), (frozen.EXECUTOR, "executor_sha256"),
                      (frozen.BASELINE, "old_checkpoint_sha256")):
        if digest(path) != policy[key]:
            raise ValueError("Preregistered comparison input changed")
    publication = load_teacher_dataset_publication(PUBLICATION)
    sft = SFTExportManifest.model_validate_json((SFT / "manifest.json").read_bytes())
    contract = frozen.mix.baseline.contract()
    if (sft.schema_version != "0.3" or sft.assistant_turn_policy != "coalesce_adjacent_assistant_messages"
            or sft.source_dataset_digest != digest(PUBLICATION / "manifest.json")
            or sft.source_dataset_id != publication.manifest.dataset_id
            or sft.chat_contract != contract or sft.chat_contract_sha256 != policy["chat_contract_sha256"]):
        raise ValueError("Source/schema/chat contract mismatch")
    for artifact in sft.artifacts:
        path = SFT / artifact.filename
        if len(path.read_bytes()) != artifact.byte_count or digest(path) != artifact.sha256:
            raise ValueError("SFT artifact integrity mismatch")
    examples = load_sft_examples(SFT / "train.jsonl")
    if len(examples) != 14 or load_sft_examples(SFT / "validation.jsonl") or publication.dataset.validation:
        raise ValueError("Only fourteen train records and empty training validation are permitted")
    for example, trajectory in zip(examples, publication.dataset.train, strict=True):
        if example.source_trajectory_id != trajectory.trajectory_id or example.source_revision != trajectory.source_revision:
            raise ValueError("SFT trajectory identity mismatch")
    plans = [load_teacher_collection_plan(module.collection.PLAN_DIR).plan for module, _, _ in frozen.mix.SOURCES]
    if plans[0].student_target != plans[1].student_target or plans[0].harness_profile != plans[1].harness_profile:
        raise ValueError("Source Student/harness mismatch")
    for plan in plans:
        verify_student_target(plan.student_target, artifact_root=ROOT)
    tasks = [assignment.task for plan in plans for assignment in plan.assignments]
    validate_teacher_dataset(publication.dataset.train, tasks=tasks,
        harness_profile=verify_harness_profile(plans[0].harness_profile, artifact_root=ROOT))
    counts = {e.source_trajectory_id: len(e.input_ids) for e in examples}
    stats = build_training_dataset_stats(publication.dataset, tasks=tasks,
        dataset_id=sft.source_dataset_id, dataset_digest=sft.source_dataset_digest,
        tokenizer_id=sft.tokenizer_id, token_counter=lambda trajectory: counts[trajectory.trajectory_id])
    labels = sum(sum(x != -100 for x in e.labels) for e in examples)
    if stats.train.exact_token_count != 9979 or labels != 3093:
        raise ValueError("Reviewed token counts changed")
    prior = TrainingRunManifest.model_validate_json((frozen.BASELINE.parent / "training-run.json").read_bytes())
    options = {key:policy[key] for key in ("epochs", "max_steps", "learning_rate", "train_batch_size",
        "gradient_accumulation_steps", "lora_rank", "lora_alpha", "lora_dropout", "target_modules")}
    options["eval_batch_size"] = prior.hyperparameters["eval_batch_size"]
    config = LoraTrainingConfig(base_model_id=policy["base_model_id"], base_model_revision=policy["base_model_revision"],
        trainer_id=prior.trainer_id, seed=policy["seed"], **options)
    if (sft.tokenizer_id != f"{config.base_model_id}@{config.base_model_revision}" or config.max_steps != 28
            or config.epochs != 2 or config.seed != 42 or not policy["fresh_base"]
            or policy["training_validation"] or not policy["no_automatic_extension"]):
        raise ValueError("Frozen training policy mismatch")
    parameters = {**options, "sft_export_id": sft.export_id, "sequence_length": policy["max_length"],
        "sft_manifest_sha256": policy["sft_manifest_sha256"], "frozen_validation_manifest_sha256": FREEZE_DIGEST,
        "evaluation_strategy": "none; independent checkpoint evaluation", "expected_effective_epochs": 2.0,
        "trainable_token_count": labels, "chat_contract_id": contract.contract_id, "chat_contract_sha256": contract.sha256(),
        "data_composition": "literal006:8 + fstring007:6, once each per epoch; fresh base, no adapter continuation",
        "scheduler_policy": policy["scheduler"], "checkpoint_selection": policy["checkpoint_selection"],
        "comparison_policy": policy["interpretation"], "model_evaluation_authorized": False,
        "authorization_scope": "Current-turn new training run only; frozen fixtures and root planning files immutable"}
    manifest = TrainingRunManifest(run_id=RUN_ID, output_capsule_id="capsule-" + RUN_ID,
        dataset_stats=stats, base_model_id=config.base_model_id, trainer_id=config.trainer_id,
        hardware_id=policy["hardware_profile_id"], random_seed=config.seed, started_at=datetime.now(UTC), hyperparameters=parameters)
    if manifest.hardware_id != plans[0].student_target.hardware.profile_id:
        raise ValueError("Hardware identity mismatch")
    return config, manifest


class ScopedRecordingBackend(recording.RecordingBackend):
    def __init__(self, before):
        self.before = before

    def train(self, **kwargs):
        directory = kwargs["run_dir"]
        save(directory / "protected-input-hashes.json", self.before)
        save(directory / "execution-authorization.json", {"training_authorized": True,
            "scope": "New 011 directory only, fixed 28 updates / two epochs; independent adapter reload",
            "model_evaluation_authorized": False, "root_planning_edits_authorized": False,
            "frozen_preparation_flags_are_historical": True})
        return super().train(**kwargs)


def verify_preservation():
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    current = protected_hashes()
    changed = [path for path, pinned in before.items() if current.get(path) != pinned]
    if changed:
        raise ValueError(f"Protected inputs changed: {changed}")
    return len(before)


def reload_only():
    if (RUN / "adapter-reload-validation.json").exists():
        raise FileExistsError("Independent reload already recorded")
    verify_preservation()
    model = reload_lora_adapter(RUN / "checkpoint.json", loader=TransformersPeftAdapterLoader())
    save(RUN / "adapter-reload-validation.json", {"run_id": RUN_ID, "independent_process": True,
        "base_model_revision_verified": True, "adapter_hashes_verified": True,
        "active_adapters": list(model.active_adapters), "loader_class": type(model).__name__,
        "protected_file_count": verify_preservation(), "generated_tokens": 0})
    print("Independent adapter reload verified; no generation or validation task executed", flush=True)
    del model
    gc.collect()


def main():
    if sys.argv[1:] == ["--reload-only"]:
        reload_only()
        return
    if sys.argv[1:] == ["--check"]:
        config, manifest = prepare()
        print(config.model_dump_json(indent=2), flush=True)
        print(manifest.dataset_stats.model_dump_json(indent=2), flush=True)
        return
    if sys.argv[1:]:
        raise ValueError("Only --check and --reload-only supported")
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing run")
    config, manifest = prepare()
    before = protected_hashes()
    print(f"Starting {RUN_ID}: fresh base, fourteen records, 28 updates / two epochs; no validation", flush=True)
    result = run_lora_training(config, training_run=manifest, train_path=SFT / "train.jsonl",
        validation_path=SFT / "validation.jsonl", output_root=RUN.parent, backend=ScopedRecordingBackend(before))
    state = json.loads((RUN / "trainer-state.json").read_text("utf-8"))
    phase = json.loads((RUN / "training-phase-metrics.json").read_text("utf-8"))
    if state["global_step"] != 28 or result.checkpoint.step != 28 or abs(state["epoch"] - 2) > 1e-8:
        raise ValueError("Authorized 28 steps / two measured epochs were not completed")
    count = verify_preservation()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count, "changed_paths": []})
    logs = [json.loads(line) for line in (RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    losses = [entry for entry in logs if "loss" in entry]
    if [entry["step"] for entry in losses] != list(range(1,29)) or any("eval_loss" in entry for entry in logs):
        raise ValueError("Missing optimizer logs or unauthorized training validation")
    summary = {**phase, "run_id": RUN_ID, "configured_epochs": config.epochs, "epoch": state["epoch"],
        "last_logged_loss": losses[-1]["loss"], "protected_file_count": count,
        "model_evaluation_started": False, "checkpoint_sha256": digest(RUN / "checkpoint.json")}
    save(RUN / "training-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
