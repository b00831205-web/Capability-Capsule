from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.learning_curve_ledger import (
    load_learning_curve_points,
)
from capability_capsule.eval.jsonl import append_jsonl
from capability_capsule.eval.records import CaseResult, DatasetSplit, RunStatus
from capability_capsule.eval.training_completion import (
    TrainingCheckpointCompleted,
    process_training_checkpoint_completion,
    process_training_checkpoint_files,
)
from capability_capsule.eval.training_provenance import (
    TrainingDatasetStats,
    TrainingRunManifest,
)
from capability_capsule.telemetry.knowledge_usage import KnowledgeUsageEvent


def training_run() -> TrainingRunManifest:
    stats = TrainingDatasetStats.model_validate(
        {
            "dataset_id": "teacher-v1",
            "dataset_digest": "a" * 64,
            "train": {
                "trajectory_count": 100,
                "message_count": 200,
                "serialized_byte_count": 10_000,
                "observable_text_character_count": 8_000,
                "tool_call_count": 50,
                "tokenizer_id": "test-tokenizer",
                "exact_token_count": 100_000,
            },
            "validation": {
                "trajectory_count": 20,
                "message_count": 40,
                "serialized_byte_count": 2_000,
                "observable_text_character_count": 1_600,
                "tool_call_count": 10,
                "tokenizer_id": "test-tokenizer",
                "exact_token_count": 20_000,
            },
            "discarded_duplicate_count": 0,
            "category_counts": {"code_search": 120},
        }
    )
    return TrainingRunManifest(
        run_id="training-run-001",
        output_capsule_id="capsule-001",
        dataset_stats=stats,
        base_model_id="base-model-v1",
        trainer_id="trainer-v1",
        hardware_id="gpu-001",
        random_seed=42,
        started_at=datetime(2026, 9, 14, tzinfo=UTC),
    )


def case_result(case_id: str, *, model_id: str = "capsule-001") -> CaseResult:
    return CaseResult(
        experiment_id="validation-001",
        run_id=f"run-{case_id}",
        case_id=case_id,
        model_id=model_id,
        condition_id="trained",
        hardware_id="gpu-001",
        repetition=1,
        cycle_position=1,
        status=RunStatus.COMPLETED,
        success=True,
        time_bounded_success=True,
        score=1.0,
        started_at=datetime(2026, 9, 14, tzinfo=UTC),
        duration_ms=100.0,
        input_tokens=100,
        output_tokens=20,
        peak_rss_mb=512.0,
        tool_call_count=1,
        invalid_tool_call_count=0,
    )


def completion(*, model_id: str = "capsule-001") -> TrainingCheckpointCompleted:
    return TrainingCheckpointCompleted(
        completed_at=datetime(2026, 9, 14, 1, 0, tzinfo=UTC),
        training_run=training_run(),
        experiment_id="validation-001",
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        results=(case_result("case-001", model_id=model_id),),
    )


def test_process_completion_records_evaluated_checkpoint(tmp_path: Path) -> None:
    ledger = tmp_path / "curves" / "cli-search.jsonl"

    recorded = process_training_checkpoint_completion(
        completion(),
        ledger_path=ledger,
    )

    assert recorded.point.checkpoint_id == "capsule-001"
    assert recorded.point.training_run_id == "training-run-001"
    assert recorded.point.cumulative_trajectory_count == 100
    assert recorded.point.success_rate == 1.0
    assert recorded.point_count == 1
    assert load_learning_curve_points(ledger) == (recorded.point,)


def test_completion_rejects_results_from_another_capsule() -> None:
    with pytest.raises(ValidationError, match="output capsule"):
        completion(model_id="another-capsule")


def test_invalid_completion_does_not_create_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "curve.jsonl"

    with pytest.raises(ValidationError):
        invalid = completion(model_id="another-capsule")
        process_training_checkpoint_completion(invalid, ledger_path=ledger)

    assert not ledger.exists()


def test_process_training_checkpoint_files_loads_standard_artifacts(
    tmp_path: Path,
) -> None:
    training_run_path = tmp_path / "training-run.json"
    results_path = tmp_path / "case-results.jsonl"
    knowledge_usage_dir = tmp_path / "knowledge-usage"
    ledger = tmp_path / "curves" / "cli-search.jsonl"

    training_run_path.write_text(
        training_run().model_dump_json(indent=2),
        encoding="utf-8",
    )
    append_jsonl(results_path, case_result("case-001"))
    knowledge_usage_dir.mkdir()
    knowledge_event = KnowledgeUsageEvent(
        capsule_id="capsule-001",
        knowledge_tree_digest="b" * 64,
        request_id="request-001",
        task_family_id="cli-search",
        required_node_ids=("node-a",),
        activated_node_ids=("node-a",),
        cache_eligible=False,
        cache_hit=False,
        cold_start=True,
    )
    (knowledge_usage_dir / "event.json").write_text(
        knowledge_event.model_dump_json(),
        encoding="utf-8",
    )

    recorded = process_training_checkpoint_files(
        training_run_path=training_run_path,
        results_path=results_path,
        knowledge_usage_dir=knowledge_usage_dir,
        ledger_path=ledger,
        completed_at=datetime(2026, 9, 14, 1, 0, tzinfo=UTC),
        experiment_id="validation-001",
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
    )

    assert recorded.point.checkpoint_id == "capsule-001"
    assert recorded.point.success_rate == 1.0
    assert recorded.point.knowledge_node_coverage == 1.0
    assert load_learning_curve_points(ledger) == (recorded.point,)


def test_invalid_training_artifact_does_not_create_ledger(
    tmp_path: Path,
) -> None:
    training_run_path = tmp_path / "training-run.json"
    results_path = tmp_path / "case-results.jsonl"
    ledger = tmp_path / "curve.jsonl"
    training_run_path.write_text("not json", encoding="utf-8")
    append_jsonl(results_path, case_result("case-001"))

    with pytest.raises(ValueError):
        process_training_checkpoint_files(
            training_run_path=training_run_path,
            results_path=results_path,
            ledger_path=ledger,
            completed_at=datetime(2026, 9, 14, 1, 0, tzinfo=UTC),
            experiment_id="validation-001",
            task_family_id="cli-search",
            evaluation_split=DatasetSplit.VALIDATION,
        )

    assert not ledger.exists()
