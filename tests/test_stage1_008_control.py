"""Recorded exposure, immutable inputs and separated evaluation evidence for 008."""

import json
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_curve_ledger import load_learning_curve_points
from capability_capsule.eval.records import CaseResult

ROOT = Path(__file__).resolve().parents[1]
ID = "stage1-codex-qwen35-2b-008-edit005-64step"
TRAINING = ROOT / "runs/training" / ID
EVALUATION = ROOT / "runs/evaluation" / ID
BASE_ID = "stage1-codex-qwen35-2b-007-edit005-32step"


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_008_protection_remains_bound_to_old_inputs_when_runner_is_reused(monkeypatch, tmp_path):
    import train_stage1_edit005_64step as control
    before = control.protected_hashes()
    monkeypatch.setattr(control.prior_eval, "RUN", tmp_path / "new-evaluation")
    monkeypatch.setattr(control.prior_eval, "TRAINING", tmp_path / "new-training")
    assert control.protected_hashes() == before
    assert f"runs/training/{BASE_ID}/checkpoint.json" in before
    assert f"runs/evaluation/{BASE_ID}/evaluation-summary.json" in before


def test_008_changes_only_step_limit_in_recorded_training_config():
    old = read(ROOT / "runs/training" / BASE_ID / "training-run.json")
    new = read(TRAINING / "training-run.json")
    for key in ("base_model_id", "trainer_id", "hardware_id", "random_seed", "dataset_stats"):
        assert new[key] == old[key]
    for key in ("epochs", "learning_rate", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
                "lora_rank", "lora_alpha", "lora_dropout", "target_modules", "sft_export_id", "evaluation_strategy"):
        assert new["hyperparameters"][key] == old["hyperparameters"][key]
    assert old["hyperparameters"]["max_steps"] == 32
    assert new["hyperparameters"]["max_steps"] == 64
    assert new["hyperparameters"]["expected_effective_epochs"] == 64 / 12
    assert "not a continuation" in new["hyperparameters"]["scheduler_policy"]
    state = read(TRAINING / "trainer-state.json")
    phase = read(TRAINING / "training-phase-metrics.json")
    assert state["global_step"] == 64
    assert abs(state["epoch"] - 64 / 12) < 1e-8
    assert phase["epoch"] == state["epoch"]
    assert phase["configured_epochs"] == 4
    assert phase["training_loss"] > 0
    steps = [json.loads(line) for line in (TRAINING / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    steps = [record for record in steps if "loss" in record]
    assert [record["step"] for record in steps] == list(range(1, 65))
    assert steps[0]["learning_rate"] == new["hyperparameters"]["learning_rate"]
    assert abs(steps[1]["learning_rate"] - 2e-4 * 63 / 64) < 1e-12
    checkpoint = read(TRAINING / "checkpoint.json")
    assert checkpoint["step"] == 64
    assert checkpoint["base_model_revision"] == read(ROOT / "runs/training" / BASE_ID / "checkpoint.json")["base_model_revision"]
    adapter = TRAINING / checkpoint["adapter_directory"]
    adapter_config = read(adapter / "adapter_config.json")
    old_adapter_config = read(ROOT / "runs/training" / BASE_ID / "adapter-final/adapter_config.json")
    for key in ("r", "lora_alpha", "lora_dropout", "bias", "task_type", "base_model_name_or_path"):
        assert adapter_config[key] == old_adapter_config[key]
    assert set(adapter_config["target_modules"]) == set(old_adapter_config["target_modules"])
    assert sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest() == checkpoint["adapter_model_sha256"]
    assert checkpoint["adapter_model_sha256"] != read(ROOT / "runs/training" / BASE_ID / "checkpoint.json")["adapter_model_sha256"]
    reload = read(TRAINING / "adapter-reload-validation.json")
    assert reload["run_id"] == ID and reload["independent_process"]
    assert reload["base_model_revision_verified"] and reload["adapter_hashes_verified"]


def test_008_did_not_change_training_sources_or_old_runs():
    for directory in (TRAINING, EVALUATION):
        evidence = read(directory / "input-preservation.json")
        hashes = read(directory / "protected-input-hashes.json")
        assert evidence["unchanged"] and evidence["changed_paths"] == []
        assert evidence["protected_file_count"] == len(hashes)
        for relative, digest in hashes.items():
            assert sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_008_fixed_and_heldout_regression_keep_007_cases_and_settings():
    for group, count in (("fixed-v3", 3), ("heldout-v1", 2)):
        directory = EVALUATION / group
        old_directory = ROOT / "runs/evaluation" / BASE_ID / group
        assert (directory / "evaluation-suite.json").read_bytes() == (old_directory / "evaluation-suite.json").read_bytes()
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results = load_jsonl(directory / "case-results.jsonl", CaseResult)
        manifest = read(directory / "run-manifest.json")
        old_manifest = read(old_directory / "run-manifest.json")
        assert len(results) == count
        for key in ("max_new_tokens", "max_tool_rounds", "do_sample", "executor_sha256", "tools_sha256", "system_prompt_sha256", "case_digest"):
            assert manifest[key] == old_manifest[key]
        assert manifest["baseline_run"] == old_directory.relative_to(ROOT).as_posix()
        assert "not newly unseen" in manifest["validation_scope"]
        assert manifest["success_count"] == sum(r.success for r in results)
        assert [r.case_id for r in results] == [c.case_id for c in suite.cases]
        points = load_learning_curve_points(directory / "learning-curve.jsonl")
        assert len(points) == 1
        assert points[0].training_run_id == ID
        assert points[0].checkpoint_id == f"{ID}-step-64"
        assert points[0].cumulative_trajectory_count == 12
        assert points[0].cumulative_token_count == 8169
        assert points[0].success_rate == sum(r.success for r in results) / count
        for result in results:
            report = read(directory / "case-reports" / f"{result.case_id}.json")
            assert report["result"] == result.model_dump(mode="json")
            if result.success:
                assert result.invalid_tool_call_count == 0
                assert all(v["passed"] for v in report["validators"])


def test_008_training_edit_probes_are_separate_from_task_scores():
    probes = read(EVALUATION / "edit-turn-probes.json")
    assert "no commands executed" in probes["scope"]
    assert probes["max_new_tokens"] == 256 and probes["do_sample"] is False
    assert len(probes["results"]) == 4
    assert {r["prompt"] for r in probes["results"]} == {"v2-training", "v3-evaluation"}
    assert {r["trajectory_id"] for r in probes["results"]} == {
        "train-edit-005-01-trajectory-001", "train-edit-005-08-trajectory-001"}
    for result in probes["results"]:
        assert result["completion"] and result["input_tokens"] > 0
        if result["target_file_matches_teacher"]:
            assert result["supported_edit"]
