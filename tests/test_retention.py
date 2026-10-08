import pytest

from capability_capsule.eval.learning_rate import LearningCurvePoint
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.retention import compare_capability_retention


def point(
    checkpoint_number: int,
    *,
    capability_id: str,
    suite_id: str,
    suite_digest: str,
    success_rate: float,
    trajectories: int | None = None,
) -> LearningCurvePoint:
    scale = checkpoint_number * 100 if trajectories is None else trajectories
    return LearningCurvePoint(
        checkpoint_id=f"capsule-c{checkpoint_number}",
        training_run_id=f"training-run-{checkpoint_number}",
        base_model_id="base-model-v1",
        capability_id=capability_id,
        evaluation_suite_id=suite_id,
        evaluation_suite_digest=suite_digest,
        task_family_id=capability_id,
        evaluation_split=DatasetSplit.VALIDATION,
        cumulative_trajectory_count=scale,
        tokenizer_id="test-tokenizer",
        cumulative_token_count=scale * 1_000,
        success_rate=success_rate,
    )


def target_curve() -> tuple[LearningCurvePoint, ...]:
    return (
        point(
            0,
            capability_id="repository-cli-navigation",
            suite_id="cli-validation-v1",
            suite_digest="a" * 64,
            success_rate=0.20,
        ),
        point(
            1,
            capability_id="repository-cli-navigation",
            suite_id="cli-validation-v1",
            suite_digest="a" * 64,
            success_rate=0.60,
        ),
    )


def anchor_curve(*, latest_success_rate: float) -> tuple[LearningCurvePoint, ...]:
    return (
        point(
            0,
            capability_id="general-anchor",
            suite_id="general-anchor-v1",
            suite_digest="b" * 64,
            success_rate=0.80,
        ),
        point(
            1,
            capability_id="general-anchor",
            suite_id="general-anchor-v1",
            suite_digest="b" * 64,
            success_rate=latest_success_rate,
        ),
    )


def test_compare_retention_reports_safe_target_improvement() -> None:
    summary = compare_capability_retention(
        target_curve(),
        anchor_curve(latest_success_rate=0.76),
        max_allowed_anchor_drop=0.05,
    )

    assert summary.base_model_id == "base-model-v1"
    assert summary.target_capability_id == "repository-cli-navigation"
    assert summary.anchor_capability_id == "general-anchor"
    assert summary.checkpoint_ids == ("capsule-c0", "capsule-c1")
    assert summary.target_gain == pytest.approx(0.40)
    assert summary.anchor_drop == pytest.approx(0.04)
    assert summary.anchor_retention_rate == pytest.approx(0.95)
    assert summary.anchor_forgetting_detected is False
    assert summary.safe_improvement is True


def test_compare_retention_detects_excessive_anchor_forgetting() -> None:
    summary = compare_capability_retention(
        target_curve(),
        anchor_curve(latest_success_rate=0.65),
        max_allowed_anchor_drop=0.05,
    )

    assert summary.anchor_drop == pytest.approx(0.15)
    assert summary.anchor_forgetting_detected is True
    assert summary.safe_improvement is False


def test_compare_retention_rejects_misaligned_checkpoints() -> None:
    anchors = list(anchor_curve(latest_success_rate=0.76))
    anchors[1] = anchors[1].model_copy(
        update={"training_run_id": "another-training-run"}
    )

    with pytest.raises(ValueError, match="aligned"):
        compare_capability_retention(target_curve(), anchors)


def test_compare_retention_requires_distinct_suites() -> None:
    with pytest.raises(ValueError, match="distinct"):
        compare_capability_retention(target_curve(), target_curve())


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_compare_retention_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        compare_capability_retention(
            target_curve(),
            anchor_curve(latest_success_rate=0.76),
            max_allowed_anchor_drop=threshold,
        )
