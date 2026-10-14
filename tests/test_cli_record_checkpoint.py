import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.eval.learning_rate import (
    LearningCurvePoint,
    summarize_task_learning_rate,
)
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.training_checkpoint_recording import (
    TrainingCheckpointRecord,
)


def recorded_checkpoint(
    ledger_path: Path,
    *,
    with_plateau: bool = False,
) -> TrainingCheckpointRecord:
    point = LearningCurvePoint(
        checkpoint_id="capsule-001",
        training_run_id="training-run-001",
        base_model_id="base-model-v1",
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_suite_digest="c" * 64,
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        cumulative_trajectory_count=100,
        tokenizer_id="test-tokenizer",
        cumulative_token_count=100_000,
        success_rate=0.75,
        knowledge_node_coverage=0.5,
    )
    if not with_plateau:
        return TrainingCheckpointRecord(
            ledger_path=ledger_path.resolve(),
            point=point,
            point_count=1,
        )

    second = point.model_copy(
        update={
            "checkpoint_id": "capsule-002",
            "training_run_id": "training-run-002",
            "cumulative_trajectory_count": 200,
            "cumulative_token_count": 200_000,
        }
    )
    third = point.model_copy(
        update={
            "checkpoint_id": "capsule-003",
            "training_run_id": "training-run-003",
            "cumulative_trajectory_count": 300,
            "cumulative_token_count": 300_000,
        }
    )
    summary = summarize_task_learning_rate((point, second, third))
    return TrainingCheckpointRecord(
        ledger_path=ledger_path.resolve(),
        point=third,
        point_count=3,
        learning_rate_summary=summary,
    )


def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    training_run = tmp_path / "training-run.json"
    results = tmp_path / "case-results.jsonl"
    evaluation_suite = tmp_path / "evaluation-suite.json"
    knowledge_usage = tmp_path / "knowledge-usage"
    ledger = tmp_path / "curves" / "cli-search.jsonl"
    training_run.touch()
    results.touch()
    evaluation_suite.touch()
    knowledge_usage.mkdir()
    return training_run, results, evaluation_suite, knowledge_usage, ledger


@pytest.mark.parametrize("json_output", [False, True])
def test_record_checkpoint_command_forwards_standard_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    training_run, results, evaluation_suite, knowledge_usage, ledger = paths(tmp_path)
    calls: list[dict[str, Any]] = []

    def process(**kwargs: Any) -> TrainingCheckpointRecord:
        calls.append(kwargs)
        return recorded_checkpoint(ledger, with_plateau=not json_output)

    monkeypatch.setattr(cli, "process_training_checkpoint_files", process)
    arguments = [
        "record-checkpoint",
        "--training-run",
        str(training_run),
        "--results",
        str(results),
        "--evaluation-suite",
        str(evaluation_suite),
        "--capability-id",
        "repository-cli-navigation",
        "--evaluation-suite-id",
        "cli-validation-v1",
        "--ledger",
        str(ledger),
        "--completed-at",
        "2026-09-15T00:00:00+00:00",
        "--experiment-id",
        "validation-001",
        "--task-family",
        "cli-search",
        "--split",
        "validation",
        "--knowledge-usage",
        str(knowledge_usage),
    ]
    if json_output:
        arguments.append("--json")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["training_run_path"] == training_run
    assert call["results_path"] == results
    assert call["evaluation_suite_path"] == evaluation_suite
    assert call["capability_id"] == "repository-cli-navigation"
    assert call["evaluation_suite_id"] == "cli-validation-v1"
    assert call["knowledge_usage_dir"] == knowledge_usage
    assert call["ledger_path"] == ledger
    assert call["completed_at"] == datetime(2026, 9, 15, tzinfo=UTC)
    assert call["experiment_id"] == "validation-001"
    assert call["task_family_id"] == "cli-search"
    assert call["evaluation_split"] is DatasetSplit.VALIDATION

    if json_output:
        payload = json.loads(result.stdout)
        assert payload["point"]["checkpoint_id"] == "capsule-001"
        assert payload["point_count"] == 1
    else:
        assert "capsule-003" in result.stdout
        assert "cli-search" in result.stdout
        assert "75.0%" in result.stdout
        assert "Regression detected: false" in result.stdout
        assert "Plateau detected: true" in result.stdout


def test_record_checkpoint_command_reports_failure_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_run, results, evaluation_suite, _, ledger = paths(tmp_path)

    def fail(**kwargs: Any) -> TrainingCheckpointRecord:
        raise ValueError("invalid checkpoint artifacts")

    monkeypatch.setattr(cli, "process_training_checkpoint_files", fail)
    result = CliRunner().invoke(
        cli.app,
        [
            "record-checkpoint",
            "--training-run",
            str(training_run),
            "--results",
            str(results),
            "--evaluation-suite",
            str(evaluation_suite),
            "--capability-id",
            "repository-cli-navigation",
            "--evaluation-suite-id",
            "cli-validation-v1",
            "--ledger",
            str(ledger),
            "--completed-at",
            "2026-09-15T00:00:00+00:00",
            "--experiment-id",
            "validation-001",
            "--task-family",
            "cli-search",
            "--split",
            "validation",
        ],
    )

    assert result.exit_code == 1
    assert not result.stdout.strip()
    assert "invalid checkpoint artifacts" in result.stderr
