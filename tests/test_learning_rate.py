import pytest
from pydantic import ValidationError

from capability_capsule.eval.learning_rate import (
    LearningCurvePoint,
    learning_curve_point_from_evaluation,
    summarize_task_learning_rate,
)
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.training_evaluation import TrainingEvaluationSummary
from capability_capsule.telemetry.knowledge_usage import (
    KnowledgeUsageEvent,
    summarize_knowledge_usage,
)


def point(
    checkpoint_id: str,
    trajectories: int,
    tokens: int,
    success_rate: float,
    coverage: float,
    *,
    task_family_id: str = "cli-search",
) -> LearningCurvePoint:
    return LearningCurvePoint(
        checkpoint_id=checkpoint_id,
        training_run_id=f"run-{checkpoint_id}",
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id=task_family_id,
        evaluation_split=DatasetSplit.VALIDATION,
        cumulative_trajectory_count=trajectories,
        tokenizer_id="test-tokenizer",
        cumulative_token_count=tokens,
        success_rate=success_rate,
        knowledge_node_coverage=coverage,
    )


def test_summarize_task_learning_rate_calculates_local_slopes() -> None:
    summary = summarize_task_learning_rate(
        (
            point("c0", 0, 0, 0.20, 0.15),
            point("c1", 100, 100_000, 0.44, 0.48),
            point("c2", 200, 200_000, 0.57, 0.66),
            point("c3", 400, 400_000, 0.61, 0.70),
        ),
        target_success_rate=0.55,
        plateau_threshold_percentage_points_per_100_trajectories=1.0,
        plateau_interval_count=2,
    )

    assert summary.task_family_id == "cli-search"
    assert summary.evaluation_split is DatasetSplit.VALIDATION
    assert summary.point_count == 4
    assert len(summary.intervals) == 3

    first, second, third = summary.intervals
    assert first.added_trajectory_count == 100
    assert first.success_rate_delta == pytest.approx(0.24)
    assert first.success_percentage_points_per_100_trajectories == pytest.approx(24.0)
    assert first.success_percentage_points_per_100k_tokens == pytest.approx(24.0)
    assert first.coverage_percentage_points_per_100_trajectories == pytest.approx(33.0)
    assert second.success_percentage_points_per_100_trajectories == pytest.approx(13.0)
    assert third.success_percentage_points_per_100_trajectories == pytest.approx(2.0)

    assert summary.target_reached is True
    assert summary.trajectories_to_target == 200
    assert summary.tokens_to_target == 200_000
    assert summary.best_success_rate == 0.61
    assert summary.latest_success_rate == 0.61
    assert summary.forgetting_rate == 0.0
    assert summary.plateau_detected is False


def test_summary_detects_forgetting_without_calling_it_a_plateau() -> None:
    summary = summarize_task_learning_rate(
        (
            point("c0", 0, 0, 0.30, 0.20),
            point("c1", 100, 100_000, 0.70, 0.60),
            point("c2", 200, 200_000, 0.62, 0.58),
        ),
        plateau_threshold_percentage_points_per_100_trajectories=10.0,
        plateau_interval_count=1,
    )

    assert summary.best_success_rate == 0.70
    assert summary.latest_success_rate == 0.62
    assert summary.forgetting_rate == pytest.approx(0.08)
    assert summary.regression_detected is True
    assert summary.plateau_detected is False


def test_summary_detects_sustained_low_positive_learning_rate() -> None:
    summary = summarize_task_learning_rate(
        (
            point("c0", 0, 0, 0.50, 0.40),
            point("c1", 100, 100_000, 0.505, 0.405),
            point("c2", 200, 200_000, 0.510, 0.410),
        ),
        plateau_threshold_percentage_points_per_100_trajectories=1.0,
        plateau_interval_count=2,
    )

    assert summary.plateau_detected is True
    assert summary.regression_detected is False


def test_summary_rejects_mixed_families_and_non_increasing_scale() -> None:
    with pytest.raises(ValueError, match="task family"):
        summarize_task_learning_rate(
            (
                point("c0", 0, 0, 0.2, 0.1),
                point("c1", 100, 100_000, 0.4, 0.3, task_family_id="greeting"),
            )
        )

    with pytest.raises(ValueError, match="strictly increase"):
        summarize_task_learning_rate(
            (
                point("c0", 100, 100_000, 0.2, 0.1),
                point("c1", 100, 200_000, 0.4, 0.3),
            )
        )


def test_learning_curve_point_rejects_partial_token_measurement() -> None:
    with pytest.raises(ValidationError):
        LearningCurvePoint(
            checkpoint_id="c0",
            training_run_id="run-c0",
            base_model_id="base-model-v1",
            capability_id="repository-cli-navigation",
            evaluation_suite_id="cli-validation-v1",
            evaluation_suite_digest="c" * 64,
            task_family_id="cli-search",
            evaluation_split=DatasetSplit.VALIDATION,
            cumulative_trajectory_count=0,
            tokenizer_id="test-tokenizer",
            cumulative_token_count=None,
            success_rate=0.2,
            knowledge_node_coverage=0.1,
        )


def evaluation_summary() -> TrainingEvaluationSummary:
    return TrainingEvaluationSummary(
        training_run_id="training-run-002",
        output_capsule_id="capsule-002",
        dataset_id="teacher-v2",
        dataset_digest="a" * 64,
        experiment_id="validation-002",
        train_trajectory_count=200,
        train_serialized_byte_count=50_000,
        train_tokenizer_id="test-tokenizer",
        train_exact_token_count=250_000,
        case_count=20,
        success_rate=0.75,
        time_bounded_success_rate=0.70,
        scored_case_count=20,
        mean_score=0.72,
        mean_duration_ms=300.0,
        mean_peak_rss_mb=512.0,
        total_input_tokens=10_000,
        total_output_tokens=2_000,
    )


def test_build_learning_curve_point_from_training_evaluation() -> None:
    knowledge_summary = summarize_knowledge_usage(
        (
            KnowledgeUsageEvent(
                capsule_id="capsule-002",
                knowledge_tree_digest="b" * 64,
                request_id="request-002",
                task_family_id="cli-search",
                required_node_ids=("required-a", "required-b"),
                activated_node_ids=("required-a", "unrelated"),
                cache_eligible=False,
                cache_hit=False,
                cold_start=True,
            ),
        )
    )

    curve_point = learning_curve_point_from_evaluation(
        evaluation_summary(),
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        knowledge_usage=knowledge_summary,
    )

    assert curve_point.schema_version == "0.2"
    assert curve_point.checkpoint_id == "capsule-002"
    assert curve_point.training_run_id == "training-run-002"
    assert curve_point.base_model_id == "base-model-v1"
    assert curve_point.capability_id == "repository-cli-navigation"
    assert curve_point.evaluation_suite_id == "cli-validation-v1"
    assert curve_point.evaluation_suite_digest == "c" * 64
    assert curve_point.cumulative_trajectory_count == 200
    assert curve_point.tokenizer_id == "test-tokenizer"
    assert curve_point.cumulative_token_count == 250_000
    assert curve_point.success_rate == 0.75
    assert curve_point.knowledge_node_coverage == 0.5


def test_unknown_required_nodes_do_not_become_zero_coverage() -> None:
    empty_knowledge_summary = summarize_knowledge_usage(())

    curve_point = learning_curve_point_from_evaluation(
        evaluation_summary(),
        checkpoint_id="checkpoint-002",
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        knowledge_usage=empty_knowledge_summary,
    )

    assert curve_point.checkpoint_id == "checkpoint-002"
    assert curve_point.knowledge_node_coverage is None
