# src/capability_capsule/training/transformers_peft.py
"""Concrete Transformers and PEFT backend for minimal LoRA training."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple

from capability_capsule.training.lora import (
    BackendCheckpoint,
    LoraTrainingConfig,
    TrainingMetric,
)
from capability_capsule.training.sft import SFTExample


class _TrainingLibraries(NamedTuple):
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    DataCollatorForSeq2Seq: Any
    Trainer: Any
    TrainingArguments: Any
    set_seed: Any
    LoraConfig: Any
    PeftModel: Any
    TaskType: Any
    get_peft_model: Any


def _load_training_libraries() -> _TrainingLibraries:
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Optional training dependencies are unavailable; "
            "install transformers, peft, accelerate, torch, and safetensors"
        ) from error

    return _TrainingLibraries(
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        DataCollatorForSeq2Seq=DataCollatorForSeq2Seq,
        Trainer=Trainer,
        TrainingArguments=TrainingArguments,
        set_seed=set_seed,
        LoraConfig=LoraConfig,
        PeftModel=PeftModel,
        TaskType=TaskType,
        get_peft_model=get_peft_model,
    )

def _disable_shadowed_dataset_namespace(trainer_class: Any) -> None:
    """Ignore a repository data directory mistaken for the datasets package."""

    trainer_module = sys.modules.get(trainer_class.__module__)
    if trainer_module is None:
        return

    dataset_module = getattr(trainer_module, "datasets", None)
    if dataset_module is None:
        return 

    if hasattr(dataset_module, "Dataset"):
        return

    trainer_module.is_datasets_available = lambda: False


class _TokenizedDataset:
    """Minimal dataset wrapper over already-tokenized SFT examples."""

    def __init__(self, examples: Sequence[SFTExample]) -> None:
        self._records = tuple(
            {
                "input_ids": list(example.input_ids),
                "attention_mask": list(example.attention_mask),
                "labels": list(example.labels),
            }
            for example in examples
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._records[index]


def _emit_training_metrics(
    history: Sequence[dict[str, Any]],
    *,
    default_learning_rate: float,
    fallback_training_loss: float | None,
    started_at: float,
    on_metric: Callable[[TrainingMetric], None],
) -> int:
    last_training_loss = fallback_training_loss
    last_learning_rate = default_learning_rate
    last_epoch = 0.0
    emitted = 0

    for record in history:
        if record.get("loss") is not None:
            last_training_loss = float(record["loss"])

        if record.get("learning_rate") is not None:
            last_learning_rate = float(record["learning_rate"])

        if record.get("epoch") is not None:
            last_epoch = float(record["epoch"])

        if record.get("eval_loss") is None:
            continue

        if last_training_loss is None:
            raise RuntimeError(
                "Trainer produced validation loss without training loss"
            )

        step = int(record.get("step", 0))
        if step < 1:
            raise RuntimeError(
                "Trainer produced validation metrics without a valid step"
            )

        on_metric(
            TrainingMetric(
                step=step,
                epoch=float(record.get("epoch", last_epoch)),
                training_loss=last_training_loss,
                validation_loss=float(record["eval_loss"]),
                learning_rate=float(
                    record.get("learning_rate", last_learning_rate)
                ),
                elapsed_seconds=max(0.0, monotonic() - started_at),
            )
        )
        emitted += 1

    return emitted


class TransformersPeftBackend:
    """Train one adapter using Transformers Trainer and PEFT."""

    def train(
        self,
        *,
        config: LoraTrainingConfig,
        train_examples: Sequence[SFTExample],
        validation_examples: Sequence[SFTExample],
        run_dir: Path,
        on_metric: Callable[[TrainingMetric], None],
    ) -> BackendCheckpoint:
        libraries = _load_training_libraries()
        _disable_shadowed_dataset_namespace(libraries.Trainer)
        libraries.set_seed(config.seed)

        run_dir = Path(run_dir)
        trainer_output = run_dir / "trainer-work"
        adapter_directory = run_dir / "adapter-final"

        tokenizer = libraries.AutoTokenizer.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
        )
        model = libraries.AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            revision=config.base_model_revision,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer must define a pad token or EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token

        if hasattr(model, "config"):
            model.config.use_cache = False

        peft_config = libraries.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type=libraries.TaskType.CAUSAL_LM,
        )
        model = libraries.get_peft_model(model, peft_config)

        collator = libraries.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
        arguments = libraries.TrainingArguments(
            output_dir=str(trainer_output),
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_strategy="steps",
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            save_strategy="no",
            seed=config.seed,
            data_seed=config.seed,
            report_to=[],
            remove_unused_columns=False,
        )

        trainer = libraries.Trainer(
            model=model,
            args=arguments,
            train_dataset=_TokenizedDataset(train_examples),
            eval_dataset=_TokenizedDataset(validation_examples),
            data_collator=collator,
            processing_class=tokenizer,
        )

        started_at = monotonic()
        training_result = trainer.train()

        fallback_loss_value = training_result.metrics.get("train_loss")
        fallback_loss = (
            float(fallback_loss_value)
            if fallback_loss_value is not None
            else None
        )

        emitted = _emit_training_metrics(
            trainer.state.log_history,
            default_learning_rate=config.learning_rate,
            fallback_training_loss=fallback_loss,
            started_at=started_at,
            on_metric=on_metric,
        )
        if emitted == 0:
            evaluation = trainer.evaluate()
            validation_loss = evaluation.get("eval_loss")
            if validation_loss is None or fallback_loss is None:
                raise RuntimeError(
                    "Trainer did not produce training and validation loss"
                )

            on_metric(
                TrainingMetric(
                    step=int(trainer.state.global_step),
                    epoch=float(config.epochs),
                    training_loss=fallback_loss,
                    validation_loss=float(validation_loss),
                    learning_rate=config.learning_rate,
                    elapsed_seconds=max(0.0, monotonic() - started_at),
                )
            )

        global_step = int(trainer.state.global_step)
        if global_step < 1:
            raise RuntimeError("Trainer completed without an optimization step")

        model.save_pretrained(
            adapter_directory,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(adapter_directory)

        return BackendCheckpoint(
            adapter_directory=adapter_directory,
            step=global_step,
        )


class TransformersPeftAdapterLoader:
    """Reload a saved PEFT adapter against its pinned base revision."""

    def load(
        self,
        *,
        base_model_id: str,
        base_model_revision: str,
        adapter_directory: Path,
    ) -> Any:
        libraries = _load_training_libraries()
        base_model = libraries.AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=base_model_revision,
        )
        model = libraries.PeftModel.from_pretrained(
            base_model,
            Path(adapter_directory),
        )

        if hasattr(model, "eval"):
            model.eval()

        return model