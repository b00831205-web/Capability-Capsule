"""Append-only persistence for task learning-curve checkpoints"""

from pathlib import Path

from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.learning_rate import(
    LearningCurvePoint,
    TaskLearningRateSummary,
    summarize_task_learning_rate,
)


def load_learning_curve_points(
        path: Path
) -> tuple[LearningCurvePoint, ...]:
    """Load and validate every persisted learning-curve point."""

    return load_jsonl(path, LearningCurvePoint)


def _validate_append(
        existing: tuple[LearningCurvePoint, ...],
        candidate: LearningCurvePoint,
) -> None:
    """Validate a candidate without modifying the existing ledger."""

    if any(
        point.checkpoint_id == candidate.checkpoint_id for point in existing
    ):
        raise ValueError(
            f"learning-curve checkpoint "
            f"{candidate.checkpoint_id!r} already exists"
        )

    if not existing:
        return

    first = existing[0]
    latest = existing[-1]

    if candidate.task_family_id != first.task_family_id:
        raise ValueError(
            "A learning-curve ledger must contain one task family"
        )

    if candidate.evaluation_split != first.evaluation_split:
        raise ValueError(
            "A learning-curve ledger must contain one evaluation split"
        )

    if candidate.tokenizer_id != first.tokenizer_id:
        raise ValueError(
            "A learning-curve ledger must use one tokenizer"
        )

    if (candidate.cumulative_trajectory_count <= latest.cumulative_trajectory_count):
        raise ValueError(
            "Cumulative trajectory count must strictly increase"
        )

    if (
        latest.cumulative_token_count is not None
        and candidate.cumulative_token_count is not None
        and candidate.cumulative_token_count <= latest.cumulative_token_count
    ):
        raise ValueError(
            "Cumulative token counts must strictly increase"
        )

def append_learning_curve_point(
        path: Path,
        point: LearningCurvePoint
) -> None:
    """Append one compatible checkpoint without rewritting prior records"""

    existing = (
        load_learning_curve_points(path)
        if path.exists() else ()
    )

    _validate_append(existing, point)
    append_jsonl(path, point)

def summarize_learning_curve_ledger(
        path: Path,
        *,
        target_success_rate: float = 0.8,
        plateau_threshold_percentage_points_per_100_trajectories: float = 1.0,
        plateau_interval_count: int = 2,
) -> TaskLearningRateSummary:
    """Load a ledger and calculate ites current learning-rate summary."""

    points = load_learning_curve_points(path)

    return summarize_task_learning_rate(
        points,
        target_success_rate = target_success_rate,
        plateau_threshold_percentage_points_per_100_trajectories= plateau_threshold_percentage_points_per_100_trajectories,
        plateau_interval_count = plateau_interval_count,
    )
