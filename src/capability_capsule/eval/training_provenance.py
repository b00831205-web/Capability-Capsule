"""Auditable links between curated data scale and capsule training runs"""

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import(
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator
)

from capability_capsule.eval.dataset_pipeline import (
    CuratedTeacherDataset,
)
from capability_capsule.eval.dataset_validation import (
    validate_teacher_dataset,
)
from capability_capsule.eval.records import TeacherTrajectory
from capability_capsule.eval.tasks import TaskCategory, TaskSpec


class SplitTrainingStats(BaseModel):
    "Measured scale of one curated training-data split."

    model_config = ConfigDict(extra ="forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    trajectory_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    serialized_byte_count: int = Field(ge=0)
    observable_text_character_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    tokenizer_id: str | None = None
    exact_token_count: int | None = Field(default=None, ge=0)

    @field_validator("tokenizer_id")
    @classmethod
    def reject_blank_tokenizer_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("tokenizer_id must not be blank")

        return value

    @model_validator(mode="after")
    def require_tokenizer_pair(self) -> Self:
        if (self.tokenizer_id is None) != (self.exact_token_count is None):
            raise ValueError(
                "tokenizer_id and exact_token_count must be provided "
                "together"
            )

        return self


class TrainingDatasetStats(BaseModel):
    """Scale and composition of a curated dataset bound to its digest."""

    model_config= ConfigDict(extra ="forbid", frozen= True)
    
    schema_version: Literal["0.1"] = "0.1"
    dataset_id: str = Field(min_length=1)
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    train: SplitTrainingStats
    validation: SplitTrainingStats
    discarded_duplicate_count: int = Field(ge=0)
    category_counts: dict[TaskCategory, int] = Field(default_factory=dict)

    @field_validator("dataset_id")
    @classmethod
    def reject_blank_dataset_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset_id must not be blank")

        return value

    @field_validator("category_counts")
    @classmethod
    def require_positive_category_counts(
        cls,
        value: dict[TaskCategory, int],
    ) -> dict[TaskCategory, int]:
        if any(count <= 0 for count in value.values()):
            raise ValueError(
                "category_counts values must be positive"
            )

        return value


class TrainingRunManifest(BaseModel):
    """Reproducible configuration for one capsule training run."""

    model_config = ConfigDict(extra = "forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    run_id: str = Field(min_length=1)
    output_capsule_id: str = Field(min_length=1)
    dataset_stats: TrainingDatasetStats
    base_model_id: str = Field(min_length=1)
    trainer_id: str = Field(min_length=1)
    hardware_id: str= Field(min_length=1)
    random_seed: int = Field(ge=0)
    started_at: datetime
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "run_id",
        "output_capsule_id",
        "base_model_id",
        "trainer_id",
        "hardware_id",
    )
    @classmethod
    def reject_blank_identifiers(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("Training-run identifiers must not be blank")

        return value

    @field_validator("started_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must include a timezone")

        return value

    @field_validator("hyperparameters")
    @classmethod
    def reject_blank_hyperparameters_names(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        if any (not name.strip() for name in value):
            raise ValueError(
                "Hyperparameter names must not be blank"
            )

        return value

    @property
    def dataset_id(self) -> str:
        """Return the immutable dataset identifier used for this run."""

        return self.dataset_stats.dataset_id

    @property
    def dataset_digest(self) -> str:
        """Return the immutable dataset digest used for this run."""

        return self.dataset_stats.dataset_digest

def _build_split_stats(
        trajectories: Sequence[TeacherTrajectory],
        *,
        tokenizer_id: str | None,
        token_counter: Callable[[TeacherTrajectory], int] | None,
) ->SplitTrainingStats:
    serialized_byte_count = 0
    message_count = 0
    observable_text_character_count = 0
    tool_call_count = 0
    exact_token_count = 0 if token_counter is not None else None

    for trajectory in trajectories:
        serialized_byte_count += len(
            trajectory.model_dump_json().encode("utf-8") + b"\n"
        )
        message_count += len(trajectory.messages)

        for message in trajectory.messages:
            observable_text_character_count += len(message.content)
            tool_call_count += len(message.tool_calls)

        if token_counter is not None:
            trajectory_token_count = token_counter(trajectory)

            if trajectory_token_count < 0:
                raise ValueError(
                    "token_counter must not return a negative value"
                )

            assert exact_token_count is not None

            exact_token_count += trajectory_token_count

    return SplitTrainingStats(
        trajectory_count = len(trajectories),
        message_count = message_count,
        serialized_byte_count= serialized_byte_count,
        observable_text_character_count=observable_text_character_count,
        tool_call_count=tool_call_count,
        tokenizer_id= tokenizer_id,
        exact_token_count=exact_token_count,
    )

def build_training_dataset_stats(
        dataset: CuratedTeacherDataset,
        *,
        tasks: Sequence[TaskSpec],
        dataset_id: str,
        dataset_digest: str,
        tokenizer_id: str| None = None,
        token_counter: Callable[[TeacherTrajectory], int] | None = None,
) -> TrainingDatasetStats:
    """Measure validated, curated data for a reproducible training run."""

    if (tokenizer_id is None) != (token_counter is None):
        raise ValueError(
            "tokenizer_id and token_counter must be provided together"
        )

    trajectories = dataset.train + dataset.validation

    validate_teacher_dataset(
        trajectories,
        tasks = tasks,
    )

    task_by_id = {
        task.task_id: task 
        for task in tasks
    }
    category_counts: dict[TaskCategory, int] = {}

    for trajectory in trajectories:
        category = task_by_id[trajectory.task_id].category
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return TrainingDatasetStats(
        dataset_id = dataset_id,
        dataset_digest= dataset_digest,
        train = _build_split_stats(
            dataset.train,
            tokenizer_id=tokenizer_id,
            token_counter=token_counter,
        ),
        validation = _build_split_stats(
            dataset.validation,
            tokenizer_id=tokenizer_id,
            token_counter = token_counter,
        ),
        discarded_duplicate_count=len(dataset.duplicates),
        category_counts = category_counts
    )