"""Record evaluated training checkpoints into learning-curve ledgers"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.eval.learning_curve_ledger import(
    append_learning_curve_point,
    load_learning_curve_points,
)
from capability_capsule.eval.learning_rate import (
    LearningCurvePoint,
    TaskLearningRateSummary,
    learning_curve_point_from_evaluation,
    summarize_task_learning_rate,
)
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.training_evaluation import TrainingEvaluationSummary
from capability_capsule.telemetry.knowledge_usage import KnowledgeUsageSummary


class TrainingCheckpointRecord(BaseModel):
    """Result of safely recording one evaluated training checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ledger_path: Path
    point: LearningCurvePoint
    point_count: int = Field(gt=0)
    learning_rate_summary: TaskLearningRateSummary | None = None

def _validate_summary_settings(
        *,
        target_success_rate: float,
        plateau_threshold_percentage_points_per_100_trajectories: float,
        plateau_interval_count: int,
) -> None:
    if not 0.0 <= target_success_rate <=1.0:
        raise ValueError(
            "target_success_rate must be between zero and one"
        )

    if (
        plateau_threshold_percentage_points_per_100_trajectories < 0.0
    ):
        raise ValueError(
            "Plateau threshold must be non-negative"
        )

    if plateau_interval_count <= 0:
        raise ValueError(
            "plateau_interval_count must be positive"
        )

def record_training_checkpoint(
        evaluation: TrainingEvaluationSummary,
        *,
        ledger_path: Path,
        base_model_id: str,
        capability_id: str,
        evaluation_suite_id: str,
        evaluation_suite_digest: str,
        task_family_id: str,
        evaluation_split: DatasetSplit,
        checkpoint_id: str | None = None,
        knowledge_usage: KnowledgeUsageSummary | None = None,
        target_success_rate: float = 0.8,
        plateau_threshold_percentage_points_per_100_trajectories: float = 1.0,
        plateau_interval_count: int =2,
) -> TrainingCheckpointRecord:
    """Convert, validate, append, and summarize one checkpoint."""

    _validate_summary_settings(
        target_success_rate= target_success_rate,
        plateau_threshold_percentage_points_per_100_trajectories= plateau_threshold_percentage_points_per_100_trajectories,
        plateau_interval_count=plateau_interval_count,
    )

    point = learning_curve_point_from_evaluation(
        evaluation,
        base_model_id = base_model_id,
        capability_id = capability_id,
        evaluation_suite_id = evaluation_suite_id,
        evaluation_suite_digest = evaluation_suite_digest,
        checkpoint_id = checkpoint_id,
        task_family_id = task_family_id,
        evaluation_split = evaluation_split,
        knowledge_usage = knowledge_usage,
    )

    existing = (
        load_learning_curve_points(ledger_path)
        if ledger_path.exists()
        else ()
    )

    prospective_points = existing + (point,)

    learning_rate_summary: TaskLearningRateSummary | None = None

    if len(prospective_points) >= 2:
        learning_rate_summary = summarize_task_learning_rate(
            prospective_points,
            target_success_rate = target_success_rate,
            plateau_threshold_percentage_points_per_100_trajectories = (
                plateau_threshold_percentage_points_per_100_trajectories
            ),
            plateau_interval_count = plateau_interval_count,
        )

    append_learning_curve_point(ledger_path, point)

    return TrainingCheckpointRecord(
        ledger_path = ledger_path.resolve(),
        point = point,
        point_count = len(prospective_points),
        learning_rate_summary= learning_rate_summary,
    )