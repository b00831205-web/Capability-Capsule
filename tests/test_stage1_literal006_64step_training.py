"""Matched exposure-control provenance, optimizer logs and immutable artifacts."""

import json
from hashlib import sha256

import pytest
import train_stage1_literal006_64step as training


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_literal006_64step_changes_only_step_limit_and_documented_metadata():
    config, expected = training.prepare()
    old = read(training.BASELINE / "training-run.json")
    new = read(training.RUN / "training-run.json")
    assert new["hyperparameters"] == expected.hyperparameters
    assert new["dataset_stats"] == old["dataset_stats"]
    for key in ("base_model_id", "trainer_id", "hardware_id", "random_seed"):
        assert new[key] == old[key]
    changed = {key for key in set(old["hyperparameters"]) | set(new["hyperparameters"])
               if old["hyperparameters"].get(key) != new["hyperparameters"].get(key)}
    assert changed == {"max_steps", "expected_effective_epochs", "scheduler_policy", "comparison_policy", "baseline_training_run_id"}
    assert config.max_steps == 64 and config.epochs == 4 and config.seed == 42
    assert new["hyperparameters"]["expected_effective_epochs"] == 8
    assert "not a continuation" in new["hyperparameters"]["scheduler_policy"]
    assert "identical LR prefix" in new["hyperparameters"]["scheduler_policy"]
    assert old["hyperparameters"]["max_steps"] == 32
    assert new["dataset_stats"]["train"]["trajectory_count"] == 8
    assert new["dataset_stats"]["train"]["exact_token_count"] == 5506
    assert new["dataset_stats"]["validation"]["trajectory_count"] == 0


def test_literal006_64step_records_all_updates_actual_epochs_and_scheduler():
    state = read(training.RUN / "trainer-state.json")
    phase = read(training.RUN / "training-phase-metrics.json")
    assert state["global_step"] == phase["step"] == 64
    assert state["epoch"] == phase["epoch"] == 8
    assert phase["configured_epochs"] == 4
    assert phase["training_loss"] > 0
    assert phase["evaluation_strategy"] == "none; independent checkpoint evaluation"
    logs = [json.loads(line) for line in (training.RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    steps = [entry for entry in logs if "loss" in entry]
    assert [entry["step"] for entry in steps] == list(range(1, 65))
    assert not any("eval_loss" in entry for entry in logs)
    assert steps[0]["learning_rate"] == 2e-4
    assert abs(steps[1]["learning_rate"] - 2e-4 * 63 / 64) < 1e-12
    assert steps[-1]["epoch"] == 8
    events = [json.loads(line) for line in (training.RUN / "trainer-process.jsonl").read_text("utf-8").splitlines()]
    assert [e["event"] for e in events] == ["started", "checkpoint_saved", "completed"]


def test_literal006_64step_adapter_hashes_and_independent_reload():
    checkpoint = read(training.RUN / "checkpoint.json")
    old = read(training.BASELINE / "checkpoint.json")
    assert checkpoint["step"] == 64 and checkpoint["run_id"] == training.RUN_ID
    assert checkpoint["base_model_revision"] == old["base_model_revision"]
    assert checkpoint["adapter_model_sha256"] != old["adapter_model_sha256"]
    adapter = training.RUN / checkpoint["adapter_directory"]
    for filename, key in (("adapter_model.safetensors", "adapter_model_sha256"), ("adapter_config.json", "adapter_config_sha256")):
        assert sha256((adapter / filename).read_bytes()).hexdigest() == checkpoint[key]
    config = read(adapter / "adapter_config.json")
    prior_config = read(training.BASELINE / "adapter-final/adapter_config.json")
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "task_type", "base_model_name_or_path"):
        assert config[key] == prior_config[key]
    assert set(config["target_modules"]) == set(prior_config["target_modules"])
    reload = read(training.RUN / "adapter-reload-validation.json")
    assert reload["run_id"] == training.RUN_ID and reload["independent_process"]
    assert reload["adapter_hashes_verified"] and reload["base_model_revision_verified"]
    assert reload["active_adapters"] == ["default"]


def test_literal006_64step_preserves_32step_evidence_and_rejects_overwrite(monkeypatch):
    before = read(training.RUN / "protected-input-hashes.json")
    evidence = read(training.RUN / "input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert evidence["protected_file_count"] == training.verify_preservation() == len(before)
    assert (training.BASELINE / "checkpoint.json").relative_to(training.ROOT).as_posix() in before
    assert "runs/evaluation/stage1-codex-qwen35-2b-009-literal006-32step/evaluation-summary.json" in before
    interrupted = training.RUN.parent / "stage1-codex-qwen35-2b-010-literal006-64step"
    assert not (interrupted / "checkpoint.json").exists()
    assert (interrupted / "optimizer-log.jsonl").relative_to(training.ROOT).as_posix() in before
    assert len((interrupted / "optimizer-log.jsonl").read_text("utf-8").splitlines()) == 2
    snapshot = (training.RUN / "checkpoint.json").read_bytes()
    monkeypatch.setattr(training.sys, "argv", ["train_stage1_literal006_64step.py"])
    with pytest.raises(FileExistsError):
        training.main()
    assert (training.RUN / "checkpoint.json").read_bytes() == snapshot
