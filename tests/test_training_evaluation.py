"""Functional tests for linking a training run to evaluation performance."""

from datetime import datetime, timezone

import pytest

from capability_capsule.eval.records import CaseResult, RunStatus
from capability_capsule.eval.training_evaluation import (
    summarize_training_evaluation,
)
from capability_capsule.eval.training_provenance import (
    TrainingDatasetStats,
    TrainingRunManifest,
)


def make_training_run() -> TrainingRunManifest:
    stats = TrainingDatasetStats.model_validate(
        {
            "dataset_id": "teacher-smoke-v1",
            "dataset_digest": "a" * 64,
            "train": {
                "trajectory_count": 50,
                "message_count": 200,
                "serialized_byte_count": 10000,
                "observable_text_character_count": 7000,
                "tool_call_count": 80,
            },
            "validation": {
                "trajectory_count": 10,
                "message_count": 40,
                "serialized_byte_count": 2000,
                "observable_text_character_count": 1400,
                "tool_call_count": 16,
            },
            "discarded_duplicate_count": 3,
            "category_counts": {"code_search": 60},
        }
    )
    return TrainingRunManifest(
        run_id="training-run-001",
        output_capsule_id="capsule-smoke-v1",
        dataset_stats=stats,
        base_model_id="base-model-v1",
        trainer_id="trainer-v1",
        hardware_id="gpu-lab-01",
        random_seed=42,
        started_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
    )


def make_case_result(
    case_id: str,
    *,
    success: bool,
    time_bounded_success: bool,
    score: float | None,
    duration_ms: float,
    peak_rss_mb: float,
    input_tokens: int,
    output_tokens: int,
    experiment_id: str = "experiment-001",
) -> CaseResult:
    return CaseResult(
        experiment_id=experiment_id,
        run_id=f"evaluation-{case_id}",
        case_id=case_id,
        model_id="capsule-smoke-v1",
        condition_id="trained",
        hardware_id="gpu-lab-01",
        repetition=1,
        cycle_position=1,
        status=(RunStatus.COMPLETED if score is not None else RunStatus.FAILED),
        success=success,
        time_bounded_success=time_bounded_success,
        score=score,
        started_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        peak_rss_mb=peak_rss_mb,
        tool_call_count=2,
        invalid_tool_call_count=0,
    )


def test_summarize_training_evaluation_links_scale_to_performance() -> None:
    training_run = make_training_run()
    results = (
        make_case_result(
            "case-001",
            success=True,
            time_bounded_success=True,
            score=1.0,
            duration_ms=100.0,
            peak_rss_mb=50.0,
            input_tokens=10,
            output_tokens=20,
        ),
        make_case_result(
            "case-002",
            success=False,
            time_bounded_success=False,
            score=None,
            duration_ms=300.0,
            peak_rss_mb=70.0,
            input_tokens=30,
            output_tokens=40,
        ),
    )

    summary = summarize_training_evaluation(
        training_run,
        experiment_id="experiment-001",
        results=results,
    )

    assert summary.training_run_id == "training-run-001"
    assert summary.dataset_id == "teacher-smoke-v1"
    assert summary.dataset_digest == "a" * 64
    assert summary.experiment_id == "experiment-001"
    assert summary.case_count == 2
    assert summary.success_rate == 0.5
    assert summary.time_bounded_success_rate == 0.5
    assert summary.scored_case_count == 1
    assert summary.mean_score == 1.0
    assert summary.mean_duration_ms == 200.0
    assert summary.mean_peak_rss_mb == 60.0
    assert summary.total_input_tokens == 40
    assert summary.total_output_tokens == 60


def test_summarize_training_evaluation_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_training_evaluation(
            make_training_run(),
            experiment_id="experiment-001",
            results=(),
        )


def test_summarize_training_evaluation_rejects_mixed_experiments() -> None:
    results = (
        make_case_result(
            "case-001",
            success=True,
            time_bounded_success=True,
            score=1.0,
            duration_ms=100.0,
            peak_rss_mb=50.0,
            input_tokens=10,
            output_tokens=20,
        ),
        make_case_result(
            "case-002",
            success=False,
            time_bounded_success=False,
            score=None,
            duration_ms=300.0,
            peak_rss_mb=70.0,
            input_tokens=30,
            output_tokens=40,
            experiment_id="experiment-002",
        ),
    )

    with pytest.raises(ValueError, match="experiment-002"):
        summarize_training_evaluation(
            make_training_run(),
            experiment_id="experiment-001",
            results=results,
        )
