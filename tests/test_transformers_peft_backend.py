"""Tests for the concrete Transformers and PEFT LoRA backend."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from capability_capsule.training.lora import LoraTrainingConfig, TrainingMetric
from capability_capsule.training.sft import SFTExample


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


def make_example(trajectory_id: str) -> SFTExample:
    return SFTExample(
        source_trajectory_id=trajectory_id,
        source_revision=f"revision-{trajectory_id}",
        input_ids=(10, 11, 12),
        attention_mask=(1, 1, 1),
        labels=(-100, 11, 12),
    )


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.saved: list[tuple[Path, bool]] = []

    def save_pretrained(
        self,
        path: Path,
        *,
        safe_serialization: bool,
    ) -> None:
        destination = Path(path)
        destination.mkdir()
        (destination / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"Qwen/Qwen3.5-0.8B"}\n',
            encoding="utf-8",
        )
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        self.saved.append((destination, safe_serialization))


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, path: Path) -> None:
        destination = Path(path)
        (destination / "tokenizer_config.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        self.saved.append(destination)


class FakeTrainer:
    latest: FakeTrainer | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.state = SimpleNamespace(
            global_step=2,
            log_history=[
                {
                    "step": 1,
                    "epoch": 0.5,
                    "loss": 1.25,
                    "learning_rate": 0.0002,
                },
                {
                    "step": 1,
                    "epoch": 0.5,
                    "eval_loss": 1.5,
                },
                {
                    "step": 2,
                    "epoch": 1.0,
                    "loss": 1.0,
                    "learning_rate": 0.0001,
                },
                {
                    "step": 2,
                    "epoch": 1.0,
                    "eval_loss": 1.2,
                },
            ],
        )
        FakeTrainer.latest = self

    def train(self) -> SimpleNamespace:
        return SimpleNamespace(metrics={"train_runtime": 0.25})


@pytest.fixture
def fake_training_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    calls = SimpleNamespace(
        model=[],
        tokenizer=[],
        lora=[],
        seeds=[],
        training_arguments=[],
        collators=[],
        peft_load=[],
    )
    base_model = FakeModel()
    adapter_model = FakeModel()
    tokenizer = FakeTokenizer()

    transformers = ModuleType("transformers")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> FakeModel:
            calls.model.append((model_id, kwargs))
            return base_model

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> FakeTokenizer:
            calls.tokenizer.append((model_id, kwargs))
            return tokenizer

    class DataCollatorForSeq2Seq:
        def __init__(self, **kwargs: Any) -> None:
            calls.collators.append(kwargs)

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            return {"features": features}

    class TrainingArguments:
        def __init__(self, **kwargs: Any) -> None:
            calls.training_arguments.append(kwargs)

    def set_seed(seed: int) -> None:
        calls.seeds.append(seed)

    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.AutoTokenizer = AutoTokenizer
    transformers.DataCollatorForSeq2Seq = DataCollatorForSeq2Seq
    transformers.Trainer = FakeTrainer
    transformers.TrainingArguments = TrainingArguments
    transformers.set_seed = set_seed

    peft = ModuleType("peft")

    class LoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls.lora.append(kwargs)

    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class PeftModel:
        @staticmethod
        def from_pretrained(
            model: FakeModel,
            adapter_directory: Path,
        ) -> FakeModel:
            calls.peft_load.append((model, Path(adapter_directory)))
            return adapter_model

    def get_peft_model(model: FakeModel, config: LoraConfig) -> FakeModel:
        return model

    peft.LoraConfig = LoraConfig
    peft.PeftModel = PeftModel
    peft.TaskType = TaskType
    peft.get_peft_model = get_peft_model

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    return SimpleNamespace(
        calls=calls,
        base_model=base_model,
        adapter_model=adapter_model,
        tokenizer=tokenizer,
    )


def test_backend_trains_tokenized_examples_and_saves_adapter(
    tmp_path: Path,
    fake_training_libraries: SimpleNamespace,
) -> None:
    from capability_capsule.training.transformers_peft import (
        TransformersPeftBackend,
    )

    metrics: list[TrainingMetric] = []
    backend = TransformersPeftBackend()
    checkpoint = backend.train(
        config=make_config(),
        train_examples=(make_example("train-001"),),
        validation_examples=(make_example("validation-001"),),
        run_dir=tmp_path,
        on_metric=metrics.append,
    )

    calls = fake_training_libraries.calls
    assert calls.seeds == [42]
    assert calls.model == [
        (
            "Qwen/Qwen3.5-0.8B",
            {"revision": "revision-001"},
        )
    ]
    assert calls.tokenizer == [
        (
            "Qwen/Qwen3.5-0.8B",
            {"revision": "revision-001"},
        )
    ]
    assert calls.lora == [
        {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    ]

    trainer = FakeTrainer.latest
    assert trainer is not None
    assert trainer.kwargs["train_dataset"][0] == {
        "input_ids": [10, 11, 12],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 11, 12],
    }
    assert trainer.kwargs["eval_dataset"][0]["labels"] == [-100, 11, 12]
    assert calls.training_arguments[0]["eval_strategy"] == "steps"
    assert calls.training_arguments[0]["save_strategy"] == "no"
    assert calls.training_arguments[0]["report_to"] == []

    assert [metric.step for metric in metrics] == [1, 2]
    assert metrics[-1].training_loss == pytest.approx(1.0)
    assert metrics[-1].validation_loss == pytest.approx(1.2)
    assert checkpoint.step == 2
    assert checkpoint.adapter_directory == tmp_path / "adapter-final"
    assert (checkpoint.adapter_directory / "adapter_model.safetensors").is_file()
    assert fake_training_libraries.tokenizer.saved == [
        tmp_path / "adapter-final"
    ]


def test_backend_supports_save_first_training_without_validation(
    tmp_path: Path,
    fake_training_libraries: SimpleNamespace,
) -> None:
    from capability_capsule.training.transformers_peft import (
        TransformersPeftBackend,
    )

    metrics: list[TrainingMetric] = []
    checkpoint = TransformersPeftBackend().train(
        config=make_config(),
        train_examples=(make_example("train-001"),),
        validation_examples=(),
        run_dir=tmp_path,
        on_metric=metrics.append,
    )

    trainer = FakeTrainer.latest
    assert trainer is not None
    assert trainer.kwargs["eval_dataset"] is None
    assert fake_training_libraries.calls.training_arguments[0][
        "eval_strategy"
    ] == "no"
    assert fake_training_libraries.calls.training_arguments[0][
        "eval_steps"
    ] is None
    assert metrics == []
    assert checkpoint.step == 2
    assert (checkpoint.adapter_directory / "adapter_model.safetensors").is_file()
    training_metrics = json.loads(
        (tmp_path / "training-phase-metrics.json").read_text("utf-8")
    )
    assert training_metrics == {
        "schema_version": "0.1",
        "run_id": tmp_path.name,
        "step": 2,
        "epoch": 1.0,
        "training_loss": 1.0,
        "evaluation_strategy": "none; independent checkpoint evaluation",
    }


def test_loader_uses_pinned_base_revision_and_adapter(
    tmp_path: Path,
    fake_training_libraries: SimpleNamespace,
) -> None:
    from capability_capsule.training.transformers_peft import (
        TransformersPeftAdapterLoader,
    )

    adapter_directory = tmp_path / "adapter-final"
    adapter_directory.mkdir()
    loader = TransformersPeftAdapterLoader()
    loaded = loader.load(
        base_model_id="Qwen/Qwen3.5-0.8B",
        base_model_revision="revision-001",
        adapter_directory=adapter_directory,
    )

    assert loaded is fake_training_libraries.adapter_model
    assert fake_training_libraries.calls.model == [
        (
            "Qwen/Qwen3.5-0.8B",
            {"revision": "revision-001"},
        )
    ]
    assert fake_training_libraries.calls.peft_load == [
        (fake_training_libraries.base_model, adapter_directory)
    ]


def test_backend_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capability_capsule.training import transformers_peft

    real_import: Callable[..., Any] = __import__

    def reject_training_libraries(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name in {"transformers", "peft"}:
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", reject_training_libraries)

    with pytest.raises(RuntimeError, match="training dependencies"):
        transformers_peft.TransformersPeftBackend().train(
            config=make_config(),
            train_examples=(make_example("train-001"),),
            validation_examples=(make_example("validation-001"),),
            run_dir=Path("unused"),
            on_metric=lambda metric: None,
        )


def test_real_tiny_qwen_training_performs_compute_and_reloads_adapter(
    tmp_path: Path,
) -> None:
    """Exercise real Torch, Transformers, and PEFT without downloading weights."""

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    from capability_capsule.training.transformers_peft import (
        TransformersPeftAdapterLoader,
        TransformersPeftBackend,
    )

    model_directory = tmp_path / "tiny-qwen3.5"
    model_directory.mkdir()

    model_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3_5ForCausalLM(model_config).save_pretrained(model_directory)

    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        **{f"token-{index}": index for index in range(4, 64)},
    }
    tokenizer_backend = Tokenizer(
        WordLevel(vocab=vocabulary, unk_token="[UNK]")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(model_directory)

    config = make_config().model_copy(
        update={
            "base_model_id": str(model_directory),
            "base_model_revision": "local-test",
            "max_steps": 1,
        }
    )
    metrics: list[TrainingMetric] = []
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    checkpoint = TransformersPeftBackend().train(
        config=config,
        train_examples=(make_example("train-001"),),
        validation_examples=(make_example("validation-001"),),
        run_dir=run_directory,
        on_metric=metrics.append,
    )

    assert checkpoint.step == 1
    assert metrics
    assert metrics[-1].training_loss >= 0
    assert metrics[-1].validation_loss >= 0
    assert (checkpoint.adapter_directory / "adapter_config.json").is_file()
    assert (
        checkpoint.adapter_directory / "adapter_model.safetensors"
    ).is_file()

    loaded = TransformersPeftAdapterLoader().load(
        base_model_id=str(model_directory),
        base_model_revision="local-test",
        adapter_directory=checkpoint.adapter_directory,
    )
    with torch.no_grad():
        output = loaded(
            input_ids=torch.tensor([[10, 11, 12]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            labels=torch.tensor([[-100, 11, 12]]),
        )

    assert torch.isfinite(output.loss)
