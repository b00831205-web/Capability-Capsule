import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.eval.learning_rate import LearningCurvePoint
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.training_checkpoint_recording import (
    TrainingCheckpointRecord,
)


def recorded_checkpoint(ledger_path: Path) -> TrainingCheckpointRecord:
    point = LearningCurvePoint(
        checkpoint_id="capsule-001",
        training_run_id="training-run-001",
        task_family_id="cli-search",
        evaluation_split=DatasetSplit.VALIDATION,
        cumulative_trajectory_count=100,
        tokenizer_id="test-tokenizer",
        cumulative_token_count=100_000,
        success_rate=0.75,
        knowledge_node_coverage=0.5,
    )
    return TrainingCheckpointRecord(
        ledger_path=ledger_path.resolve(),
        point=point,
        point_count=1,
    )


def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    training_run = tmp_path / "training-run.json"
    results = tmp_path / "case-results.jsonl"
    knowledge_usage = tmp_path / "knowledge-usage"
    ledger = tmp_path / "curves" / "cli-search.jsonl"
    training_run.touch()
    results.touch()
    knowledge_usage.mkdir()
    return training_run, results, knowledge_usage, ledger


@pytest.mark.parametrize("json_output", [False, True])
def test_record_checkpoint_command_forwards_standard_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    training_run, results, knowledge_usage, ledger = paths(tmp_path)
    calls: list[dict[str, Any]] = []

    def process(**kwargs: Any) -> TrainingCheckpointRecord:
        calls.append(kwargs)
        return recorded_checkpoint(ledger)

    monkeypatch.setattr(cli, "process_training_checkpoint_files", process)
    arguments = [
        "record-checkpoint",
        "--training-run",
        str(training_run),
        "--results",
        str(results),
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
        assert "capsule-001" in result.stdout
        assert "cli-search" in result.stdout
        assert "75.0%" in result.stdout


def test_record_checkpoint_command_reports_failure_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_run, results, _, ledger = paths(tmp_path)

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
