import json
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.telemetry.report import TelemetryReport


def _report(output_dir: Path) -> TelemetryReport:
    return TelemetryReport(
        output_dir=output_dir.resolve(),
        event_count=4,
        success_count=3,
        error_count=1,
        success_rate=0.75,
        mean_duration_ms=125.0,
        p95_duration_ms=250.0,
        total_question_chars=80,
        total_source_count=9,
        error_types={"ConnectError": 1},
        capsule_build_ids=(uuid4(),),
        generation_models=("qwen3.5:4b",),
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_report_command_displays_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    output_dir = tmp_path / "sessions"
    output_dir.mkdir()
    calls: list[Path] = []

    def summarize(path: Path) -> TelemetryReport:
        calls.append(path)
        return _report(path)

    monkeypatch.setattr(cli, "summarize_telemetry", summarize)
    arguments = ["report", str(output_dir)]
    if json_output:
        arguments.append("--json")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert calls == [output_dir]
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["event_count"] == 4
        assert payload["success_rate"] == 0.75
        assert payload["error_types"] == {"ConnectError": 1}
    else:
        assert "4" in result.stdout
        assert "75" in result.stdout
        assert "125" in result.stdout
        assert "250" in result.stdout
        assert "ConnectError" in result.stdout


def test_report_command_reports_invalid_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "sessions"
    output_dir.mkdir()

    def fail(path: Path) -> TelemetryReport:
        raise ValueError("invalid telemetry event")

    monkeypatch.setattr(cli, "summarize_telemetry", fail)
    result = CliRunner().invoke(cli.app, ["report", str(output_dir)])

    assert result.exit_code != 0
    assert "invalid telemetry event" in result.output
