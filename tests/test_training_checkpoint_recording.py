from pathlib import Path

import pytest

from capability_capsule.eval.learning_curve_ledger import (
    load_learning_curve_points,
)
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.training_checkpoint_recording import (
    record_training_checkpoint,
)
from capability_capsule.eval.training_evaluation import TrainingEvaluationSummary


def evaluation(
    checkpoint: str,
    trajectories: int,
    tokens: int,
    success_rate: float,
) -> TrainingEvaluationSummary:
    return TrainingEvaluationSummary(
        training_run_id=f"run-{checkpoint}",
        output_capsule_id=checkpoint,
        dataset_id=f"dataset-{checkpoint}",
        dataset_digest="a" * 64,
        experiment_id="validation-001",
        train_trajectory_count=trajectories,
        train_serialized_byte_count=max(1, trajectories * 100),
        train_tokenizer_id="test-tokenizer",
        train_exact_token_count=tokens,
        case_count=20,
        success_rate=success_rate,
        time_bounded_success_rate=success_rate,
        scored_case_count=20,
        mean_score=success_rate,
        mean_duration_ms=100.0,
        mean_peak_rss_mb=256.0,
        total_input_tokens=1_000,
        total_output_tokens=200,
    )


def test_record_training_checkpoint_builds_ledger_then_summary(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "learning-curves" / "cli-search.jsonl"

    first = record_training_checkpoint(
        evaluation("capsule-c0", 0, 0, 0.20),
        ledger_path=ledger,
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
    )

    assert first.point.checkpoint_id == "capsule-c0"
    assert first.point.base_model_id == "base-model-v1"
    assert first.point.capability_id == "repository-cli-navigation"
    assert first.point.evaluation_suite_id == "cli-validation-v1"
    assert first.point.evaluation_suite_digest == "c" * 64
    assert first.point_count == 1
    assert first.learning_rate_summary is None
    assert len(load_learning_curve_points(ledger)) == 1

    second = record_training_checkpoint(
        evaluation("capsule-c1", 100, 100_000, 0.44),
        ledger_path=ledger,
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        target_success_rate=0.40,
    )

    assert second.point.checkpoint_id == "capsule-c1"
    assert second.point_count == 2
    assert second.learning_rate_summary is not None
    assert second.learning_rate_summary.target_reached is True
    assert second.learning_rate_summary.trajectories_to_target == 100
    assert second.learning_rate_summary.intervals[0].success_rate_delta == pytest.approx(
        0.24
    )


def test_invalid_checkpoint_does_not_modify_existing_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "curve.jsonl"
    baseline = evaluation("capsule-c0", 0, 0, 0.20)
    record_training_checkpoint(
        baseline,
        ledger_path=ledger,
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
    )
    before = ledger.read_bytes()

    with pytest.raises(ValueError):
        record_training_checkpoint(
            baseline,
            ledger_path=ledger,
            base_model_id="base-model-v1",
            capability_id="repository-cli-navigation",
            evaluation_suite_id="cli-validation-v1",
            evaluation_suite_digest="c" * 64,
            task_family_id="cli-search",
            evaluation_split=DatasetSplit.VALIDATION,
        )

    assert ledger.read_bytes() == before


def test_invalid_summary_settings_do_not_append_checkpoint(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "curve.jsonl"
    record_training_checkpoint(
        evaluation("capsule-c0", 0, 0, 0.20),
        ledger_path=ledger,
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
    )
    before = ledger.read_bytes()

    with pytest.raises(ValueError, match="Plateau threshold"):
        record_training_checkpoint(
            evaluation("capsule-c1", 100, 100_000, 0.44),
            ledger_path=ledger,
            base_model_id="base-model-v1",
            capability_id="repository-cli-navigation",
            evaluation_suite_id="cli-validation-v1",
            evaluation_suite_digest="c" * 64,
            task_family_id="cli-search",
            evaluation_split=DatasetSplit.VALIDATION,
            plateau_threshold_percentage_points_per_100_trajectories=-1.0,
        )

    assert ledger.read_bytes() == before
