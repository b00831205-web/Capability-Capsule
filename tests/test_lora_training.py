"""Tests for the minimal append-only LoRA training entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest
from capability_capsule.training.lora import (
    BackendCheckpoint,
    LoraTrainingConfig,
    SavedAdapterCheckpoint,
    TrainerProcessEvent,
    TrainingMetric,
    reload_lora_adapter,
    run_lora_training,
)

from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.training_provenance import (
    SplitTrainingStats,
    TrainingDatasetStats,
    TrainingRunManifest,
)
from capability_capsule.training.sft import SFTExample


def make_training_run() -> TrainingRunManifest:
    return TrainingRunManifest(
        run_id="smoke-lora-001",
        output_capsule_id="capsule-smoke-001",
        dataset_stats=TrainingDatasetStats(
            dataset_id="qwen-smoke-001",
            dataset_digest="a" * 64,
            train=SplitTrainingStats(
                trajectory_count=1,
                message_count=4,
                serialized_byte_count=100,
                observable_text_character_count=50,
                tool_call_count=1,
                tokenizer_id="Qwen/Qwen3.5-0.8B@revision-001",
                exact_token_count=6,
            ),
            validation=SplitTrainingStats(
                trajectory_count=1,
                message_count=4,
                serialized_byte_count=100,
                observable_text_character_count=50,
                tool_call_count=1,
                tokenizer_id="Qwen/Qwen3.5-0.8B@revision-001",
                exact_token_count=6,
            ),
            discarded_duplicate_count=0,
        ),
        base_model_id="Qwen/Qwen3.5-0.8B",
        trainer_id="transformers-peft-lora-v1",
        hardware_id="hardware-training-smoke-001",
        random_seed=42,
        started_at=datetime(2026, 9, 16, tzinfo=UTC),
        hyperparameters={
            "epochs": 1,
            "max_steps": 2,
            "learning_rate": 0.0002,
            "lora_rank": 8,
        },
    )


def make_config() -> LoraTrainingConfig:
    return LoraTrainingConfig(
        base_model_id="Qwen/Qwen3.5-0.8B",
        base_model_revision="revision-001",
        trainer_id="transformers-peft-lora-v1",
        seed=42,
        epochs=1,
        max_steps=2,
        learning_rate=0.0002,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )


def write_sft_split(path: Path, trajectory_id: str) -> None:
    append_jsonl(
        path,
        SFTExample(
            source_trajectory_id=trajectory_id,
            source_revision=f"revision-{trajectory_id}",
            input_ids=(10, 11, 12),
            attention_mask=(1, 1, 1),
            labels=(-100, 11, 12),
        ),
    )


class SuccessfulBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def train(
        self,
        *,
        config: LoraTrainingConfig,
        train_examples: Sequence[SFTExample],
        validation_examples: Sequence[SFTExample],
        run_dir: Path,
        on_metric: Callable[[TrainingMetric], None],
    ) -> BackendCheckpoint:
        self.calls.append(
            {
                "config": config,
                "train_examples": tuple(train_examples),
                "validation_examples": tuple(validation_examples),
                "run_dir": run_dir,
            }
        )
        on_metric(
            TrainingMetric(
                step=1,
                epoch=0.5,
                training_loss=1.25,
                validation_loss=1.5,
                learning_rate=0.0002,
                elapsed_seconds=0.25,
            )
        )
        adapter_dir = run_dir / "adapter-final"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"Qwen/Qwen3.5-0.8B"}\n',
            encoding="utf-8",
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
        return BackendCheckpoint(
            adapter_directory=adapter_dir,
            step=2,
        )


class FailingBackend:
    def train(self, **kwargs: Any) -> BackendCheckpoint:
        raise RuntimeError("synthetic out of memory")


class RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []

    def load(
        self,
        *,
        base_model_id: str,
        base_model_revision: str,
        adapter_directory: Path,
    ) -> object:
        self.calls.append(
            (base_model_id, base_model_revision, adapter_directory)
        )
        return {"loaded": True}


def test_training_run_persists_metrics_checkpoint_and_reload_identity(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "sft" / "train.jsonl"
    validation_path = tmp_path / "sft" / "validation.jsonl"
    write_sft_split(train_path, "train-001")
    write_sft_split(validation_path, "validation-001")
    backend = SuccessfulBackend()

    result = run_lora_training(
        make_config(),
        training_run=make_training_run(),
        train_path=train_path,
        validation_path=validation_path,
        output_root=tmp_path / "runs",
        backend=backend,
    )

    run_dir = tmp_path / "runs" / "smoke-lora-001"
    assert result.run_directory == run_dir.resolve()
    assert result.adapter_directory == (run_dir / "adapter-final").resolve()
    assert result.checkpoint.step == 2
    assert (run_dir / "training-run.json").is_file()
    assert (run_dir / "checkpoint.json").is_file()
    assert backend.calls[0]["train_examples"][0].source_trajectory_id == "train-001"
    assert (
        backend.calls[0]["validation_examples"][0].source_trajectory_id
        == "validation-001"
    )

    events = load_jsonl(
        run_dir / "trainer-process.jsonl",
        TrainerProcessEvent,
    )
    assert [event.event for event in events] == [
        "started",
        "metric",
        "checkpoint_saved",
        "completed",
    ]
    assert events[1].training_loss == pytest.approx(1.25)
    assert events[1].validation_loss == pytest.approx(1.5)

    loader = RecordingLoader()
    loaded = reload_lora_adapter(
        run_dir / "checkpoint.json",
        loader=loader,
    )

    assert loaded == {"loaded": True}
    assert loader.calls == [
        (
            "Qwen/Qwen3.5-0.8B",
            "revision-001",
            (run_dir / "adapter-final").resolve(),
        )
    ]


def test_training_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    (output_root / "smoke-lora-001").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="smoke-lora-001"):
        run_lora_training(
            make_config(),
            training_run=make_training_run(),
            train_path=tmp_path / "unused-train.jsonl",
            validation_path=tmp_path / "unused-validation.jsonl",
            output_root=output_root,
            backend=SuccessfulBackend(),
        )


def test_training_failure_is_logged_and_reraised(tmp_path: Path) -> None:
    train_path = tmp_path / "sft" / "train.jsonl"
    validation_path = tmp_path / "sft" / "validation.jsonl"
    write_sft_split(train_path, "train-001")
    write_sft_split(validation_path, "validation-001")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_lora_training(
            make_config(),
            training_run=make_training_run(),
            train_path=train_path,
            validation_path=validation_path,
            output_root=tmp_path / "runs",
            backend=FailingBackend(),
        )

    events = load_jsonl(
        tmp_path / "runs" / "smoke-lora-001" / "trainer-process.jsonl",
        TrainerProcessEvent,
    )
    assert [event.event for event in events] == ["started", "failed"]
    assert events[-1].error_type == "RuntimeError"
    assert not (
        tmp_path / "runs" / "smoke-lora-001" / "checkpoint.json"
    ).exists()


def test_training_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    mismatched = make_training_run().model_copy(
        update={"base_model_id": "different/model"}
    )

    with pytest.raises(ValueError, match="base model"):
        run_lora_training(
            make_config(),
            training_run=mismatched,
            train_path=tmp_path / "unused-train.jsonl",
            validation_path=tmp_path / "unused-validation.jsonl",
            output_root=tmp_path / "runs",
            backend=SuccessfulBackend(),
        )


def test_stage1_codex_training_run_has_verified_checkpoint_and_reload() -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / "training" / "stage1-codex-qwen35-2b-001"

    training_run = TrainingRunManifest.model_validate_json(
        (run_dir / "training-run.json").read_bytes()
    )
    checkpoint = SavedAdapterCheckpoint.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    events = load_jsonl(
        run_dir / "trainer-process.jsonl",
        TrainerProcessEvent,
    )
    recovery = json.loads((run_dir / "recovery.json").read_text("utf-8"))
    reload_validation = json.loads(
        (run_dir / "adapter-reload-validation.json").read_text("utf-8")
    )

    assert training_run.run_id == "stage1-codex-qwen35-2b-001"
    assert training_run.dataset_id == "stage1-codex-001"
    assert training_run.dataset_digest == (
        "989582326bb6e9c0e34a3afe3c3dd847ea1bbc3ed322f6366af042f4b27ad3cb"
    )
    assert training_run.dataset_stats.train.exact_token_count == 4803
    assert training_run.dataset_stats.validation.exact_token_count == 2734
    assert training_run.hyperparameters["sft_export_id"] == (
        "stage1-codex-001-qwen35-2b-v2"
    )
    assert training_run.hyperparameters["max_steps"] == 8

    assert checkpoint.step == 8
    assert checkpoint.base_model_revision == (
        "15852e8c16360a2fea060d615a32b45270f8a8fc"
    )
    assert checkpoint.adapter_config_sha256 == (
        "04a7470914b5d504d3e281d4ed1f4f92d45866001a878d59e87fa702996f0821"
    )
    assert checkpoint.adapter_model_sha256 == (
        "68505fb0cc54421c04e05b58e75f948d2b6d02ab6d69e19cd84681f2a7918cc5"
    )
    adapter_dir = run_dir / checkpoint.adapter_directory
    assert sha256((adapter_dir / "adapter_config.json").read_bytes()).hexdigest() == (
        checkpoint.adapter_config_sha256
    )
    assert sha256(
        (adapter_dir / "adapter_model.safetensors").read_bytes()
    ).hexdigest() == checkpoint.adapter_model_sha256

    assert [event.event for event in events] == [
        "started",
        "failed",
        "started",
        "metric",
        "checkpoint_saved",
        "completed",
    ]
    assert events[1].error_type == "WSLProcessTerminated"
    assert events[3].step == 8
    assert events[3].training_loss == pytest.approx(1.5090183913707733)
    assert events[3].validation_loss == pytest.approx(1.0871495008468628)
    assert recovery["recovery_policy"]["evaluation_strategy"] == (
        "single validation pass after final optimization step"
    )
    assert (run_dir / "terminal.log").is_file()
    assert (run_dir / "terminal-retry.log").is_file()

    assert reload_validation["base_model_revision_verified"] is True
    assert reload_validation["adapter_hashes_verified"] is True
    assert reload_validation["loader_class"] == "PeftModelForCausalLM"
    assert reload_validation["active_adapters"] == ["default"]


def test_stage1_powershell_recovery_checkpoint_is_saved_and_reloaded() -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs/training/stage1-codex-qwen35-2b-002-recovery-004"
    training_run = TrainingRunManifest.model_validate_json(
        (run_dir / "training-run.json").read_bytes()
    )
    checkpoint = SavedAdapterCheckpoint.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    events = load_jsonl(run_dir / "trainer-process.jsonl", TrainerProcessEvent)
    training_metrics = json.loads(
        (run_dir / "training-phase-metrics.json").read_text("utf-8")
    )
    validation_metrics = json.loads(
        (run_dir / "validation-phase-metrics.json").read_text("utf-8")
    )

    assert training_run.dataset_id == "stage1-codex-powershell-002"
    assert training_run.hyperparameters["sft_export_id"] == (
        "stage1-codex-powershell-002-qwen35-2b-v1"
    )
    assert training_run.hyperparameters["evaluation_strategy"] == (
        "separate process after adapter save"
    )
    assert checkpoint.step == 16
    assert checkpoint.adapter_config_sha256 == (
        "604cd23943abad5ee3c58a175969f0f4e013c8b9695debbd56594de4f13e1723"
    )
    assert checkpoint.adapter_model_sha256 == (
        "8a4c9043f95cb40765b1c17bc427f8cfd7b33892e57057bd20f42df2bc665cc1"
    )
    assert [event.event for event in events] == [
        "started",
        "checkpoint_saved",
        "completed",
    ]
    assert training_metrics["training_loss"] == pytest.approx(1.2680502831935883)
    assert validation_metrics["validation_loss"] == pytest.approx(1.022344172000885)
    assert validation_metrics["model_class"] == "PeftModelForCausalLM"
