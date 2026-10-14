from pathlib import Path

import pytest

from capability_capsule.eval.learning_curve_ledger import (
    append_learning_curve_point,
    load_learning_curve_points,
    summarize_learning_curve_ledger,
)
from capability_capsule.eval.learning_rate import LearningCurvePoint
from capability_capsule.eval.records import DatasetSplit


def point(
    checkpoint_id: str,
    trajectories: int,
    tokens: int,
    success_rate: float,
    *,
    task_family_id: str = "cli-search",
    split: DatasetSplit = DatasetSplit.VALIDATION,
    base_model_id: str = "base-model-v1",
    capability_id: str = "repository-cli-navigation",
    evaluation_suite_id: str = "cli-validation-v1",
    evaluation_suite_digest: str = "c" * 64,
) -> LearningCurvePoint:
    return LearningCurvePoint(
        checkpoint_id=checkpoint_id,
        training_run_id=f"run-{checkpoint_id}",
        base_model_id=base_model_id,
        capability_id=capability_id,
        evaluation_suite_id=evaluation_suite_id,
        evaluation_suite_digest=evaluation_suite_digest,
        task_family_id=task_family_id,
        evaluation_split=split,
        cumulative_trajectory_count=trajectories,
        tokenizer_id="test-tokenizer",
        cumulative_token_count=tokens,
        success_rate=success_rate,
        knowledge_node_coverage=success_rate,
    )


def test_append_learning_curve_round_trips_and_summarizes(tmp_path: Path) -> None:
    ledger = tmp_path / "metrics" / "cli-search-validation.jsonl"
    first = point("c0", 0, 0, 0.20)
    second = point("c1", 100, 100_000, 0.44)

    append_learning_curve_point(ledger, first)
    append_learning_curve_point(ledger, second)

    assert load_learning_curve_points(ledger) == (first, second)
    assert ledger.read_text(encoding="utf-8").endswith("\n")

    summary = summarize_learning_curve_ledger(
        ledger,
        target_success_rate=0.40,
    )

    assert summary.point_count == 2
    assert summary.target_reached is True
    assert summary.trajectories_to_target == 100
    assert summary.intervals[0].success_rate_delta == pytest.approx(0.24)


def test_duplicate_checkpoint_is_rejected_without_modifying_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "curve.jsonl"
    append_learning_curve_point(ledger, point("c0", 0, 0, 0.20))
    before = ledger.read_bytes()

    with pytest.raises(ValueError, match="checkpoint"):
        append_learning_curve_point(
            ledger,
            point("c0", 100, 100_000, 0.50),
        )

    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "candidate",
    [
        point("c2", 50, 200_000, 0.50),
        point("c2", 200, 50_000, 0.50),
        point("c2", 200, 200_000, 0.50, task_family_id="greeting"),
        point("c2", 200, 200_000, 0.50, split=DatasetSplit.TEST),
        point("c2", 200, 200_000, 0.50, base_model_id="base-model-v2"),
        point("c2", 200, 200_000, 0.50, capability_id="debugging"),
        point("c2", 200, 200_000, 0.50, evaluation_suite_id="cli-validation-v2"),
        point("c2", 200, 200_000, 0.50, evaluation_suite_digest="d" * 64),
    ],
)
def test_incompatible_append_is_rejected_without_modifying_ledger(
    tmp_path: Path,
    candidate: LearningCurvePoint,
) -> None:
    ledger = tmp_path / "curve.jsonl"
    append_learning_curve_point(ledger, point("c0", 0, 0, 0.20))
    append_learning_curve_point(ledger, point("c1", 100, 100_000, 0.40))
    before = ledger.read_bytes()

    with pytest.raises(ValueError):
        append_learning_curve_point(ledger, candidate)

    assert ledger.read_bytes() == before


def test_summary_requires_at_least_two_persisted_points(tmp_path: Path) -> None:
    ledger = tmp_path / "curve.jsonl"
    append_learning_curve_point(ledger, point("c0", 0, 0, 0.20))

    with pytest.raises(ValueError, match="two learning-curve points"):
        summarize_learning_curve_ledger(ledger)
