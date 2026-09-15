"""Minimal auditable LoRA training orchestration and adapter reload"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from capability_capsule.eval.jsonl import append_jsonl
from capability_capsule.eval.training_provenance import TrainingRunManifest
from capability_capsule.training.sft import SFTExample, load_sft_examples


class LoraTrainingConfig(BaseModel):
    """Fixed minimal cconfiguration for one LoRA run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    trainer_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    epochs: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    train_batch_size: int = Field(ge=1)
    eval_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0, lt=1)
    target_modules: tuple[str, ...] = Field(min_length=1)

    @field_validator("base_model_id", "base_model_revision", "trainer_id")
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LoRA identifiers must not be blank")

        return value

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, value: tuple[str, ...])-> tuple[str, ...]:
        if any(not module.strip() for module in value):
            raise ValueError("LoRA target modules must not be blank")

        if len(set(value)) != len(value):
            raise ValueError("LoRA target modules must be unique")

        return value

class TrainingMetric(BaseModel):
    """One measured point emitted by a training backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=1)
    epoch: float = Field(ge=0)
    training_loss: float = Field(ge=0)
    validation_loss: float = Field(ge=0)
    learning_rate: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)

class TrainerProcessEvent(BaseModel):
    """One append-only training process event"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    event: Literal["started", "metric", "checkpoint_saved","completed","failed"]
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str= Field(min_length=1)
    step: int | None = Field(default=None, ge=0)
    epoch: float | None = Field(default=None, ge=0)
    training_loss: float | None = Field(default=None, ge=0)
    validation_loss: float | None = Field(default = None, ge=0)
    learning_rate: float | None = Field(default=None, ge=0)
    elapsed_seconds: float | None = Field(default=None, ge=0)
    checkpoint_path: str | None = None
    error_type: str | None = None

    @field_validator("run_id")
    @classmethod
    def reject_blank_run_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must not be blank")

        return value

    @field_validator("checkpoint_path", "error_type")
    @classmethod
    def reject_blank_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Optional event text must not be blank")

        return value

    @model_validator(mode="after")
    def validate_event_shape(self) -> Self:
        if self.event == "metric":
            required = (
                self.step,
                self.epoch,
                self.training_loss,
                self.validation_loss,
                self.learning_rate,
                self.elapsed_seconds,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "Metric events require all metric measurements"
                )

        if (self.event == "checkpoint_saved" and (self.step is None or self.checkpoint_path is None)):
            raise ValueError(
                "Checkpoint events require step and checkpoint_path"
            )

        if self.event == "failed":
            if self.error_type is None:
                raise ValueError("Failed events require error_type")

        elif self.error_type is not None:
            raise ValueError(
                "Only failed events may include error_type"
            )

        return self

class BackendCheckpoint(BaseModel):
    """Checkpoint returned by the concrete training backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_directory: Path
    step: int = Field(ge=1)

class SavedAdapterCheckpoint(BaseModel):
    """Portable identity and hashes for one saved adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    run_id: str = Field(min_length=1)
    output_capsule_id: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    adapter_directory: str = Field(min_length=1)
    step: int = Field(ge=1)
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

class LoraTrainingResult(BaseModel):
    """Artifacts produced by one completed training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_directory: Path
    process_log_path: Path
    adapter_directory: Path
    checkpoint: SavedAdapterCheckpoint

class LoraTrainingBackend(Protocol):
    """Backend boundary implemented later with Transformers and PEFT"""

    def train(
            self,
            *,
            config: LoraTrainingConfig,
            train_examples: Sequence[SFTExample],
            validation_examples: Sequence[SFTExample],
            run_dir: Path,
            on_metric: Callable[[TrainingMetric], None],
    ) -> BackendCheckpoint:
        """Train and save one adapter inside run_dir"""

class LoraAdapterLoader(Protocol):
    """Backend boundary for loading a saved adapter"""

    def load(
            self,
            *,
            base_model_id: str,
            base_model_revision: str,
            adapter_directory: Path,
    ) -> Any:
        """Load the pinned base model and adapter."""

def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _write_json_exclusive(path: Path, record: BaseModel) -> None:
    encoded = record.model_dump_json(indent=2).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)

def _validate_run_identity(
        config: LoraTrainingConfig,
        training_run: TrainingRunManifest,
) -> None:
    if config.base_model_id != training_run.base_model_id:
        raise ValueError(
            "LoRA config base model does not match training manifest"
        )

    if config.trainer_id != training_run.trainer_id:
        raise ValueError(
            "LoRA config trainer does not match training manifest"
        )

    if config.seed != training_run.random_seed:
        raise ValueError(
            "LoRA config seed does not match training manifest"
        )

    expected = {
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank, 
    }

    for name, value in expected.items():
        recorded = training_run.hyperparameters.get(name)
        if recorded is not None and recorded != value:
            raise ValueError(
                f"LoRA hyperparameter {name!r} does not match training manifest"
            )


def _validate_dataset_counts(
        training_run: TrainingRunManifest,
        train_examples: Sequence[SFTExample],
        validation_examples: Sequence[SFTExample],
) -> None:
    if not train_examples:
        raise ValueError("LoRA training split must not be empty")

    if not validation_examples:
        raise ValueError("LoRA validation split must not be empty")

    if len(train_examples) != training_run.dataset_stats.train.trajectory_count:
        raise ValueError("SFT train record count does not match training manifest")

    if len(validation_examples) != training_run.dataset_stats.validation.trajectory_count:
        raise ValueError("SFT validation record count does not match training manifest")


def _resolve_adapter_directory(run_dir: Path, backend_checkpoint: BackendCheckpoint) -> Path:

    adapter_directory = backend_checkpoint.adapter_directory.resolve(strict=True)

    resolved_run_dir = run_dir.resolve()
    try:
        adapter_directory.relative_to(resolved_run_dir)
    except ValueError as error:
        raise ValueError(
            "Training backend saved adapter outside the run directory"
        ) from error

    if not adapter_directory.is_dir():
        raise NotADirectoryError(adapter_directory)

    return adapter_directory

def _build_saved_checkpoint(
        config: LoraTrainingConfig,
        training_run: TrainingRunManifest,
        run_dir: Path,
        backend_checkpoint: BackendCheckpoint,
) -> tuple[SavedAdapterCheckpoint, Path]:
    adapter_directory = _resolve_adapter_directory(
        run_dir, backend_checkpoint
    )

    adapter_config = adapter_directory / "adapter_config.json"
    adapter_model = adapter_directory / "adapter_model.safetensors"

    if not adapter_config.is_file():
        raise FileNotFoundError(adapter_config)

    if not adapter_model.is_file():
        raise FileNotFoundError(adapter_model)

    relative_adapter = adapter_directory.relative_to(run_dir.resolve())

    checkpoint = SavedAdapterCheckpoint(
        run_id= training_run.run_id,
        output_capsule_id= training_run.output_capsule_id,
        base_model_id= config.base_model_id,
        base_model_revision= config.base_model_revision,
        adapter_directory= relative_adapter.as_posix(),
        step= backend_checkpoint.step,
        adapter_config_sha256= _sha256_file(adapter_config),
        adapter_model_sha256= _sha256_file(adapter_model),
    )
    return checkpoint, adapter_directory

def run_lora_training(
        config: LoraTrainingConfig,
        *,
        training_run: TrainingRunManifest,
        train_path: Path,
        validation_path: Path,
        output_root: Path,
        backend: LoraTrainingBackend,
) -> LoraTrainingResult:
    """Run one non-overwritting LoRA job with append-only evidence."""

    _validate_run_identity(config, training_run)

    output_root = Path(output_root)
    run_dir = output_root / training_run.run_id

    if run_dir.exists():
        raise FileExistsError(f"Training run {training_run.run_id!r} already exists")

    train_examples = load_sft_examples(Path(train_path))
    validation_examples = load_sft_examples(Path(validation_path))
    _validate_dataset_counts(training_run, train_examples, validation_examples)

    output_root.mkdir(parents=True, exist_ok = True)
    run_dir.mkdir()
    process_log_path = run_dir / "trainer-process.jsonl"

    _write_json_exclusive(
        run_dir / "training-run.json", training_run
    )
    append_jsonl(
        process_log_path,
        TrainerProcessEvent(
            event="started",
            run_id = training_run.run_id,
            step = 0,
        ),
    )

    def on_metric(metric: TrainingMetric) -> None:
        append_jsonl(
            process_log_path,
            TrainerProcessEvent(
                event="metric",
                run_id= training_run.run_id,
                step = metric.step,
                epoch = metric.epoch,
                training_loss = metric.training_loss,
                validation_loss= metric.validation_loss,
                learning_rate = metric.learning_rate,
                elapsed_seconds= metric.elapsed_seconds,
            ),
        )

    try:
        backend_checkpoint = backend.train(
            config = config,
            train_examples=train_examples,
            validation_examples= validation_examples,
            run_dir= run_dir,
            on_metric= on_metric,
        )
        checkpoint, adapter_directory = _build_saved_checkpoint(
            config,
            training_run,
            run_dir,
            backend_checkpoint,
        )
        checkpoint_path = run_dir / "checkpoint.json"
        _write_json_exclusive(checkpoint_path, checkpoint)

        append_jsonl(
            process_log_path,
            TrainerProcessEvent(
                event="checkpoint_saved",
                run_id = training_run.run_id,
                step = checkpoint.step,
                checkpoint_path = checkpoint.adapter_directory,
            ),
        )
        append_jsonl(
            process_log_path,
            TrainerProcessEvent(
                event= "completed",
                run_id= training_run.run_id,
                step=checkpoint.step,
            ),
        )
    except Exception as error:
        append_jsonl(
            process_log_path,
            TrainerProcessEvent(
                event="failed",
                run_id=training_run.run_id,
                error_type = type(error).__name__,
            ),
        )
        raise
    return LoraTrainingResult(
        run_directory = run_dir.resolve(),
        process_log_path = process_log_path.resolve(),
        adapter_directory= adapter_directory,
        checkpoint = checkpoint,
    )

def reload_lora_adapter(
        checkpoint_path: Path,
        *,
        loader: LoraAdapterLoader,
) -> Any:
    """Verify adapter hashes and reload it with the pinned base revision."""

    checkpoint_path = Path(checkpoint_path).resolve(strict = True)
    checkpoint = SavedAdapterCheckpoint.model_validate_json(checkpoint_path.read_bytes())
    run_dir = checkpoint_path.parent
    adapter_directory = (run_dir / checkpoint.adapter_directory).resolve(strict=True)

    try:
        adapter_directory.relative_to(run_dir)
    except ValueError as error:
        raise ValueError(
            "Checkpoint adapter path escapes the run directory"
        ) from error

    adapter_config = adapter_directory / "adapter_config.json"
    adapter_model = adapter_directory / "adapter_model.safetensors"

    if _sha256_file(adapter_config) != checkpoint.adapter_config_sha256:
        raise ValueError("Adapter configuration SHA-256 mismatch")

    if _sha256_file(adapter_model) != checkpoint.adapter_model_sha256:
        raise ValueError("Adapter model SHA-256 mismatch")

    return loader.load(
        base_model_id = checkpoint.base_model_id,
        base_model_revision = checkpoint.base_model_revision,
        adapter_directory= adapter_directory,
    )