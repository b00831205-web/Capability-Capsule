"""Measure task learning rates across reproducible training checkpoints."""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)

from capability_capsule.eval.training_evaluation import (
    TrainingEvaluationSummary,
)
from capability_capsule.telemetry.knowledge_usage import (
    KnowledgeUsageSummary,
)

from capability_capsule.eval.records import DatasetSplit


class LearningCurvePoint(BaseModel):
    """One task-family evaluation result at a training checkpoint."""

    model_config = ConfigDict(extra= "forbid", frozen= True)

    schema_version: Literal["0.2"] = "0.2"
    checkpoint_id: str = Field(min_length=1)
    training_run_id: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    evaluation_suite_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_family_id: str = Field(min_length=1)
    evaluation_split: DatasetSplit
    cumulative_trajectory_count: int = Field(ge = 0)
    tokenizer_id: str | None = None
    cumulative_token_count: int | None = Field(default=None, ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    knowledge_node_coverage: float | None = Field(
        default = None,
        ge= 0.0,
        le = 1.0
    )

    @field_validator(
        "checkpoint_id",
        "training_run_id",
        "task_family_id",
        "tokenizer_id",
        "base_model_id",
        "capability_id",
        "evaluation_suite_id",
    )
    @classmethod
    def reject_blank_identifier(
        cls,
        value: str | None
    ) -> str | None:
        if value is not None and  not value.strip():
            raise ValueError("Learning-curve identifier must not be blank")

        return value


    @model_validator(mode="after")
    def validate_token_measurement(self) -> Self:
        if (self.tokenizer_id is None) != (
            self.cumulative_token_count is None
        ):
            raise ValueError(
                "tokenizer_id and cumulative_token_count "
                "must be provided together"
            )

        return self


class LearningRateInterval(BaseModel):
    """Capability change between two consecutive checkpoint"""

    model_config = ConfigDict(extra="forbid", frozen= True)

    from_checkpoint_id: str = Field(min_length=1)
    to_checkpoint_id: str = Field(min_length=1)
    added_trajectory_count: int = Field(gt=0)
    added_token_count: int | None = Field(default=None, gt=0)
    success_rate_delta: float = Field(ge=-1.0, le=1.0)
    success_percentage_points_per_100_trajectories: float
    success_percentage_points_per_100k_tokens: float | None = None
    coverage_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    coverage_percentage_points_per_100_trajectories: float | None = None

class TaskLearningRateSummary(BaseModel):
    """Learning-rate summary for one task family and evaluation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.2"] = "0.2"
    task_family_id: str = Field(min_length=1)
    evaluation_split: DatasetSplit
    tokenizer_id: str | None = None
    point_count: int = Field(ge=2)
    points: tuple[LearningCurvePoint, ...] = Field(min_length=2)
    intervals: tuple[LearningRateInterval, ...] = Field(min_length=1)
    target_success_rate: float = Field(ge=0.0, le=1.0)
    target_reached: bool
    trajectories_to_target: int | None = Field(default=None, ge=0)
    tokens_to_target: int | None = Field(default=None, ge=0)
    best_success_rate: float = Field(ge=0.0, le=1.0)
    latest_success_rate: float = Field(ge=0.0, le=1.0)
    forgetting_rate: float = Field(ge=0.0, le=1.0)
    regression_detected: bool
    plateau_detected: bool
    plateau_threshold_percentage_points_per_100_trajectories: float = Field(
        ge=0.0
    )
    plateau_interval_count: int = Field(gt=0)
    base_model_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    evaluation_suite_digest: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

def _build_interval(
        previous: LearningCurvePoint,
        current: LearningCurvePoint
) -> LearningRateInterval:
    added_trajectory_count = (
        current.cumulative_trajectory_count - previous.cumulative_trajectory_count
    )

    if added_trajectory_count <= 0:
        raise ValueError(
            "Cumulative trajectory counts must strictly increase"
        )

    added_token_count: int | None = None
    success_per_100k_tokens: float | None = None

    if (previous.cumulative_token_count is not None and current.cumulative_token_count is not None):
        added_token_count = (current.cumulative_token_count - previous.cumulative_token_count)

        if added_token_count <= 0:
            raise ValueError(
                "Cumulative token counts must strictly increase"
            )

    success_rate_delta = current.success_rate - previous.success_rate

    success_per_100_trajectories = (
        success_rate_delta *100.0 *100.0 / added_trajectory_count
    )

    if added_token_count is not None:
        success_per_100k_tokens = (
            success_rate_delta *100.0 *100_000.0 /added_token_count
        )

    coverage_delta: float|None = None
    coverage_per_100_trajectories: float | None = None

    if (
        previous.knowledge_node_coverage is not None
        and current.knowledge_node_coverage is not None
    ):
        coverage_delta = (
            current.knowledge_node_coverage - previous.knowledge_node_coverage
        )
        coverage_per_100_trajectories = (
            coverage_delta *100.0 *100.0 /added_trajectory_count
        )


    return LearningRateInterval(
        from_checkpoint_id= previous.checkpoint_id,
        to_checkpoint_id = current.checkpoint_id,
        added_trajectory_count= added_trajectory_count,
        added_token_count= added_token_count,
        success_rate_delta= success_rate_delta,
        success_percentage_points_per_100_trajectories= success_per_100_trajectories,
        success_percentage_points_per_100k_tokens= success_per_100k_tokens,
        coverage_delta= coverage_delta,
        coverage_percentage_points_per_100_trajectories= coverage_per_100_trajectories,
    )

def learning_curve_point_from_evaluation(
        evaluation: TrainingEvaluationSummary,
        *,
        task_family_id: str,
        evaluation_split: DatasetSplit,
        checkpoint_id: str | None = None,
        knowledge_usage: KnowledgeUsageSummary | None = None,
        base_model_id: str,
        capability_id: str,
        evaluation_suite_id: str,
        evaluation_suite_digest: str,
) -> LearningCurvePoint:
    """Convert one training evaluation into a learning-curve point."""

    knowledge_node_coverage: float | None = None

    if (
        knowledge_usage is not None and knowledge_usage.required_node_count > 0
    ):
        knowledge_node_coverage = (
            knowledge_usage.knowledge_node_coverage
        )

    return LearningCurvePoint(
        checkpoint_id = (
            checkpoint_id if checkpoint_id is not None else evaluation.output_capsule_id
        ),
        training_run_id = evaluation.training_run_id,
        task_family_id=task_family_id,
        evaluation_split=evaluation_split,
        cumulative_trajectory_count= evaluation.train_trajectory_count,
        tokenizer_id = evaluation.train_tokenizer_id,
        cumulative_token_count=evaluation.train_exact_token_count,
        success_rate=evaluation.success_rate,
        knowledge_node_coverage=knowledge_node_coverage,
        base_model_id=base_model_id,
        capability_id=capability_id,
        evaluation_suite_id=evaluation_suite_id,
        evaluation_suite_digest=evaluation_suite_digest,
    )

def summarize_task_learning_rate(
        points: Sequence[LearningCurvePoint],
        *,
        target_success_rate: float = 0.8,
        plateau_threshold_percentage_points_per_100_trajectories: float = 1.0,
        plateau_interval_count: int = 2
) -> TaskLearningRateSummary:
    """Summarize task learning, forgetting, and plateau behavior."""

    if len(points) < 2:
        raise ValueError(
            "At least two learning-curve points are required"
        )

    if not 0.0 <= target_success_rate <= 1.0:
        raise ValueError(
            "target_success_rate must be between zero and one"
        )

    if plateau_threshold_percentage_points_per_100_trajectories < 0.0:
        raise ValueError(
            "Plateau threshold must be non-negative"
        )

    if plateau_interval_count <= 0:
        raise ValueError(
            "plateau_interval_count must be positive"
        )

    ordered_points = tuple(points)
    first = ordered_points[0]

    if any(
        point.task_family_id != first. task_family_id
        for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must belong to one task family"
        )

    if any(
        point.evaluation_split != first.evaluation_split for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must use one evaluation split"
        )
    if any(
        point.base_model_id != first.base_model_id
        for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must use one base model"
        )

    if any(
        point.capability_id != first.capability_id
        for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must measure one capability"
        )

    if any(
        point.evaluation_suite_id
        != first.evaluation_suite_id
        for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must use one evaluation suite"
        )

    if any(
        point.evaluation_suite_digest
        != first.evaluation_suite_digest
        for point in ordered_points
    ):
        raise ValueError(
            "All learning-curve points must use unchanged evaluation cases"
        )

    checkpoint_ids = [
        point.checkpoint_id for point in ordered_points
    ]

    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError(
            "Learning-curve checkpoint identifiers must be unique"
        )

    measured_tokens = [
        point.cumulative_token_count is not None for point in ordered_points
    ]

    if any(measured_tokens) and not all(measured_tokens):
        raise ValueError(
            "Token measurement must be present at every checkpoint "
            "or omitted from every checkpoint"
        )

    tokenizer_ids = {
        point.tokenizer_id for point in ordered_points if point.tokenizer_id is not None
    }

    if len(tokenizer_ids) > 1:
        raise ValueError(
            "All learning-curve points must use one tokenizer"
        )

    intervals = tuple(
        _build_interval(previous, current)
        for previous, current in zip(
            ordered_points,
            ordered_points[1:],
        )
    )

    target_point = next(
        (point for point in ordered_points if point.success_rate >= target_success_rate), None
    )

    best_success_rate = max(
        point.success_rate for point in ordered_points
    )
    latest_success_rate = ordered_points[-1].success_rate
    forgetting_rate = max(
        0.0,
        best_success_rate - latest_success_rate,
    )
    regression_detected = forgetting_rate > 0.0
    plateau_detected = False

    if len(intervals) >= plateau_interval_count:
        recent_intervals = intervals[-plateau_interval_count:]
        recent_rates = [
            interval.success_percentage_points_per_100_trajectories for interval in recent_intervals
        ]
        plateau_detected = all(
            0.0 <= rate <= plateau_threshold_percentage_points_per_100_trajectories for rate in recent_rates
        )

    tokenizer_id = (
        next(iter(tokenizer_ids)) if tokenizer_ids else None
    )

    return TaskLearningRateSummary(
        task_family_id = first.task_family_id,
        evaluation_split = first.evaluation_split,
        tokenizer_id = tokenizer_id,
        point_count = len(ordered_points),
        points = ordered_points,
        intervals = intervals,
        target_success_rate= target_success_rate,
        target_reached = target_point is not None,
        trajectories_to_target = (
            target_point.cumulative_trajectory_count
            if target_point is not None
            else None
        ),
        tokens_to_target=(
            target_point.cumulative_token_count if target_point is not None else None
        ),
        best_success_rate = best_success_rate,
        latest_success_rate= latest_success_rate,
        forgetting_rate= forgetting_rate,
        regression_detected= regression_detected,
        plateau_threshold_percentage_points_per_100_trajectories=(
            plateau_threshold_percentage_points_per_100_trajectories
        ),
        plateau_interval_count=plateau_interval_count,
        plateau_detected=plateau_detected,
        base_model_id=first.base_model_id,
        capability_id=first.capability_id,
        evaluation_suite_id=first.evaluation_suite_id,
        evaluation_suite_digest=first.evaluation_suite_digest,
    )
