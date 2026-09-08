from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest

from capability_capsule.config import TelemetryConfig
from capability_capsule.telemetry.writer import write_run_telemetry


def _write_event(
    directory: Path,
    *,
    build_id: UUID,
    duration_ms: float,
    status: Literal["success", "error"],
    source_count: int,
    error_type: str | None = None,
) -> None:
    write_run_telemetry(
        TelemetryConfig(enabled=True, output_dir=directory),
        capsule_path=directory.parent / "flight.zip",
        capsule_build_id=build_id,
        generation_model="qwen3.5:4b",
        duration_ms=duration_ms,
        question_chars=20,
        source_count=source_count,
        status=status,
        error_type=error_type,
    )


def test_summarize_telemetry_calculates_runtime_metrics(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.report", fromlist=["summarize_telemetry"])
    directory = tmp_path / "sessions"
    first_build = uuid4()
    second_build = uuid4()
    _write_event(
        directory,
        build_id=first_build,
        duration_ms=100.0,
        status="success",
        source_count=2,
    )
    _write_event(
        directory,
        build_id=first_build,
        duration_ms=300.0,
        status="success",
        source_count=3,
    )
    _write_event(
        directory,
        build_id=second_build,
        duration_ms=50.0,
        status="error",
        source_count=0,
        error_type="ConnectError",
    )

    report = module.summarize_telemetry(directory)

    assert report.output_dir == directory.resolve()
    assert report.event_count == 3
    assert report.success_count == 2
    assert report.error_count == 1
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.mean_duration_ms == pytest.approx(150.0)
    assert report.p95_duration_ms == pytest.approx(300.0)
    assert report.total_question_chars == 60
    assert report.total_source_count == 5
    assert report.error_types == {"ConnectError": 1}
    assert set(report.capsule_build_ids) == {first_build, second_build}
    assert report.generation_models == ("qwen3.5:4b",)


def test_summarize_empty_directory_returns_zero_report(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.report", fromlist=["summarize_telemetry"])
    directory = tmp_path / "empty"
    directory.mkdir()

    report = module.summarize_telemetry(directory)

    assert report.event_count == 0
    assert report.success_count == 0
    assert report.error_count == 0
    assert report.success_rate == 0.0
    assert report.mean_duration_ms == 0.0
    assert report.p95_duration_ms == 0.0
    assert report.error_types == {}


def test_summarize_telemetry_rejects_missing_or_invalid_input(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.report", fromlist=["summarize_telemetry"])

    with pytest.raises((OSError, ValueError)):
        module.summarize_telemetry(tmp_path / "missing")

    directory = tmp_path / "sessions"
    directory.mkdir()
    (directory / "broken.json").write_text("not json", encoding="utf-8")

    with pytest.raises((OSError, ValueError)):
        module.summarize_telemetry(directory)
