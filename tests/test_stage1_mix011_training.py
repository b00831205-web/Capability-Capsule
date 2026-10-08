"""Frozen 28-step configuration, process evidence and independent reload."""

import json

import pytest
import train_stage1_mix011_28step as training


def read(name):
    return json.loads((training.RUN / name).read_text("utf-8"))


def test_mix011_prepare_matches_frozen_budget_and_excludes_validation():
    config, manifest = training.prepare()
    assert config.max_steps == 28 and config.epochs == 2 and config.seed == 42
    assert config.learning_rate == 2e-4
    assert config.train_batch_size == config.gradient_accumulation_steps == config.eval_batch_size == 1
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (8,16,0.05)
    assert config.target_modules == ("q_proj", "v_proj")
    assert manifest.dataset_stats.train.trajectory_count == 14
    assert manifest.dataset_stats.train.exact_token_count == 9979
    assert manifest.dataset_stats.validation.trajectory_count == 0
    assert manifest.hyperparameters["trainable_token_count"] == 3093
    assert not manifest.hyperparameters["model_evaluation_authorized"]
    recorded = read("training-run.json")
    assert recorded["hyperparameters"] == manifest.hyperparameters
    assert recorded["dataset_stats"] == manifest.dataset_stats.model_dump(mode="json")
    assert read("execution-authorization.json")["training_authorized"]
    assert not read("execution-authorization.json")["root_planning_edits_authorized"]


def test_mix011_all_28_updates_and_two_epochs_recorded_without_validation():
    state, phase, summary = read("trainer-state.json"), read("training-phase-metrics.json"), read("training-summary.json")
    assert state["global_step"] == phase["step"] == summary["step"] == 28
    assert state["epoch"] == summary["epoch"] == summary["configured_epochs"] == 2
    assert phase["evaluation_strategy"] == "none; independent checkpoint evaluation"
    assert phase["training_loss"] > 0 and not summary["model_evaluation_started"]
    logs = [json.loads(line) for line in (training.RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    updates = [entry for entry in logs if "loss" in entry]
    assert [entry["step"] for entry in updates] == list(range(1,29))
    assert not any("eval_loss" in entry for entry in logs)
    assert updates[0]["learning_rate"] == 2e-4
    assert abs(updates[1]["learning_rate"] - 2e-4 * 27/28) < 1e-12
    assert updates[-1]["epoch"] == 2 and summary["last_logged_loss"] == updates[-1]["loss"]
    events = [json.loads(line) for line in (training.RUN / "trainer-process.jsonl").read_text("utf-8").splitlines()]
    assert [event["event"] for event in events] == ["started", "checkpoint_saved", "completed"]


def test_mix011_checkpoint_hashes_and_independent_reload_without_generation():
    checkpoint = read("checkpoint.json")
    assert checkpoint["step"] == 28 and checkpoint["run_id"] == training.RUN_ID
    assert checkpoint["base_model_revision"] == training.frozen.mix.baseline.REVISION
    adapter = training.RUN / checkpoint["adapter_directory"]
    assert training.digest(adapter / "adapter_model.safetensors") == checkpoint["adapter_model_sha256"]
    assert training.digest(adapter / "adapter_config.json") == checkpoint["adapter_config_sha256"]
    reload = read("adapter-reload-validation.json")
    assert reload["run_id"] == training.RUN_ID and reload["independent_process"]
    assert reload["adapter_hashes_verified"] and reload["base_model_revision_verified"]
    assert reload["active_adapters"] == ["default"] and reload["generated_tokens"] == 0


def test_mix011_preserves_frozen_validation_old_runs_and_root_docs_and_refuses_overwrite(monkeypatch):
    protected, evidence = read("protected-input-hashes.json"), read("input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert training.verify_preservation() == evidence["protected_file_count"] == len(protected)
    assert "MVP_ROADMAP.md" in protected and "PENDING_CONFIGURATION_OPTIMIZATIONS.md" in protected
    assert (training.frozen.OUTPUT / "freeze-manifest.json").relative_to(training.ROOT).as_posix() in protected
    assert training.frozen.BASELINE.relative_to(training.ROOT).as_posix() in protected
    before = (training.RUN / "checkpoint.json").read_bytes()
    monkeypatch.setattr(training.sys, "argv", ["train.py"])
    with pytest.raises(FileExistsError):
        training.main()
    assert (training.RUN / "checkpoint.json").read_bytes() == before
