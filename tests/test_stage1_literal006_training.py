"""Training provenance/integrity only; no checkpoint task evaluation."""

import json
from hashlib import sha256

import pytest
import train_stage1_literal006_32step as training


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_literal006_training_uses_reviewed_sft_and_fixed_fresh_base_config():
    config, expected = training.prepare()
    recorded = read(training.RUN / "training-run.json")
    assert recorded["run_id"] == training.RUN_ID
    assert recorded["dataset_stats"] == expected.dataset_stats.model_dump(mode="json")
    assert recorded["dataset_stats"]["train"]["trajectory_count"] == 8
    assert recorded["dataset_stats"]["train"]["exact_token_count"] == 5506
    assert recorded["dataset_stats"]["validation"]["trajectory_count"] == 0
    assert recorded["hyperparameters"] == expected.hyperparameters
    assert config.max_steps == 32 and config.epochs == 4 and config.seed == 42
    assert config.learning_rate == 2e-4
    assert config.train_batch_size == config.gradient_accumulation_steps == 1
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (8, 16, 0.05)
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    assert config.base_model_revision == training.source.REVISION
    assert "fresh base model" in recorded["hyperparameters"]["data_composition"]
    assert "not a single-variable control" in recorded["hyperparameters"]["comparison_policy"]


def test_literal006_training_completes_measured_epochs_and_records_all_steps():
    state = read(training.RUN / "trainer-state.json")
    phase = read(training.RUN / "training-phase-metrics.json")
    assert state["global_step"] == 32
    assert state["epoch"] == phase["epoch"] == phase["configured_epochs"] == 4
    assert phase["training_loss"] > 0
    assert phase["evaluation_strategy"] == "none; independent checkpoint evaluation"
    logs = [json.loads(line) for line in (training.RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    steps = [entry for entry in logs if "loss" in entry]
    assert [entry["step"] for entry in steps] == list(range(1, 33))
    assert not any("eval_loss" in entry for entry in logs)
    assert steps[0]["learning_rate"] == 2e-4
    assert abs(steps[1]["learning_rate"] - 2e-4 * 31 / 32) < 1e-12
    events = [json.loads(line) for line in (training.RUN / "trainer-process.jsonl").read_text("utf-8").splitlines()]
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    assert any(e["event"] == "checkpoint_saved" and e["step"] == 32 for e in events)
    assert not any(e["event"] in {"failed", "metric"} for e in events)


def test_literal006_training_saves_reloadable_hash_verified_adapter():
    checkpoint = read(training.RUN / "checkpoint.json")
    assert checkpoint["run_id"] == training.RUN_ID and checkpoint["step"] == 32
    assert checkpoint["base_model_revision"] == training.source.REVISION
    adapter = training.RUN / checkpoint["adapter_directory"]
    assert sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest() == checkpoint["adapter_model_sha256"]
    assert sha256((adapter / "adapter_config.json").read_bytes()).hexdigest() == checkpoint["adapter_config_sha256"]
    adapter_config = read(adapter / "adapter_config.json")
    assert (adapter_config["r"], adapter_config["lora_alpha"], adapter_config["lora_dropout"]) == (8, 16, 0.05)
    assert set(adapter_config["target_modules"]) == {"q_proj", "v_proj"}
    reload = read(training.RUN / "adapter-reload-validation.json")
    assert reload["run_id"] == training.RUN_ID and reload["independent_process"]
    assert reload["adapter_hashes_verified"] and reload["base_model_revision_verified"]
    assert reload["active_adapters"] == ["default"]


def test_literal006_training_preserves_inputs_and_refuses_existing_run(monkeypatch):
    before = read(training.RUN / "protected-input-hashes.json")
    preservation = read(training.RUN / "input-preservation.json")
    assert preservation["unchanged"] and preservation["changed_paths"] == []
    assert preservation["protected_file_count"] == training.verify_preservation() == len(before)
    assert (training.SFT / "train.jsonl").relative_to(training.ROOT).as_posix() in before
    assert "runs/training/stage1-codex-qwen35-2b-008-edit005-64step/checkpoint.json" in before
    snapshot = (training.RUN / "checkpoint.json").read_bytes()
    monkeypatch.setattr(training.sys, "argv", ["train_stage1_literal006_32step.py"])
    with pytest.raises(FileExistsError):
        training.main()
    assert (training.RUN / "checkpoint.json").read_bytes() == snapshot
