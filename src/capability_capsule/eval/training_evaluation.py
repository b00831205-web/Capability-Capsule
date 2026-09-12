"""Comparable evaluation summaries for reproducible capsule training runs."""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capability_capsule.eval.records import CaseResult
from capability_capsule.eval.training_provenance import (
    TrainingRunManifest
)


class TrainingEvaluationSummary(BaseModel):
    """One performance point linking training scale to evaluation outcomes."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    training_run_id: str = Field(min_length =1)
    output_capsule_id: str = Field(min_length =1)
    dataset_id: str = Field(min_length=1)
    dataset_digest: str = Field(pattern = r"^[0-9a-f]{64}$")
    experiment_id: str = Field(min_length=1)
    train_trajectory_count: int = Field(ge=0)
    train_serialized_byte_count: int = Field(ge=0)
    train_tokenizer_id: str | None = None
    train_exact_token_count: int | None = Field(default = None, ge = 0)
    case_count: int = Field(ge=1)
    success_rate: float = Field(ge = 0.0, le = 1.0)
    time_bounded_success_rate: float = Field(ge=0.0, le =1.0)
    scored_case_count: int = Field(ge=0)
    mean_score: float | None = Field(default = None, ge = 0.0, le= 1.0)
    mean_duration_ms: float = Field(ge=0.0)
    mean_peak_rss_mb: float = Field(ge = 0.0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge = 0)

    @field_validator(
        "training_run_id",
        "output_capsule_id",
        "dataset_id",
        "experiment_id",
        "train_tokenizer_id",
    )
    @classmethod
    def reject_blank_identifiers(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Evaluation identifiers must not be blank")

        return value

    @model_validator(mode="after")
    def validate_aggregate_paris(self) -> Self:
        if (self.train_tokenizer_id is None) != (
            self.train_exact_token_count is None
        ):
            raise ValueError(
                "train_tokenizer_id and train_exact_token_count "
                "must be provided together"
            )

        if (self.scored_case_count == 0) != (self.mean_score is None):
            raise ValueError(
                "mean_score is required exactly when scored_case_count "
                "is positive"
            )

        return self

def summarize_training_evaluation(
        training_run: TrainingRunManifest,
        *,
        experiment_id: str,
        results: Sequence[CaseResult]
) -> TrainingEvaluationSummary:
    """Aggregate one experiment into a point on a capability-growth curve."""

    if not results:
        raise ValueError(
            "Training evaluation requires at least one case result"
        )

    for result in results:
        if result.experiment_id != experiment_id:
            raise ValueError(
                f"Case result {result.case_id!r} belongs to "
                f"experiment {result.experiment_id!r}, not "
                f"{experiment_id!r}"
            )

    case_count = len(results)
    success_count = sum(result.success for result in results)
    time_bounded_success_count = sum(
        result.time_bounded_success
        for result in results
    )

    scores = [
        result.score
        for result in results
        if result.score is not None
    ]

    train_stats = training_run.dataset_stats.train

    return TrainingEvaluationSummary(
        training_run_id = training_run.run_id,
        output_capsule_id= training_run.output_capsule_id,
        dataset_id = training_run.dataset_id,
        dataset_digest = training_run.dataset_digest,
        experiment_id = experiment_id,
        train_trajectory_count=  train_stats.trajectory_count,
        train_serialized_byte_count=train_stats.serialized_byte_count,
        train_tokenizer_id= train_stats.tokenizer_id,
        train_exact_token_count= train_stats.exact_token_count,
        case_count = case_count,
        success_rate = success_count / case_count,
        time_bounded_success_rate= time_bounded_success_count / case_count,
        scored_case_count= len(scores),
        mean_score = (sum(scores) / len(scores) if scores else None),
        mean_duration_ms = (
            sum(result.duration_ms for result in results) / case_count
        ),
        mean_peak_rss_mb=(
            sum(result.peak_rss_mb for result in results) / case_count
        ),
        total_input_tokens= sum(result.input_tokens for result in results),
        total_output_tokens= sum(result.output_tokens for result in results),
    )

