import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.runtime.readiness import ReadinessReport


def _report(capsule_path: Path, *, missing: tuple[str, ...] = ()) -> ReadinessReport:
    return ReadinessReport(
        capsule_path=capsule_path.resolve(),
        capsule_build_id=uuid4(),
        ollama_version="0.33.2",
        required_models=("qwen3.5:4b", "nomic-embed-text"),
        installed_models=("nomic-embed-text:latest", "qwen3.5:4b"),
        missing_models=missing,
        ready=not missing,
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_doctor_reports_ready_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    capsule_path = tmp_path / "flight.zip"
    capsule_path.write_bytes(b"placeholder")
    calls: list[Path] = []

    def check(path: Path) -> ReadinessReport:
        calls.append(path)
        return _report(path)

    monkeypatch.setattr(cli, "check_capsule_readiness", check)
    arguments = ["doctor", str(capsule_path)]
    if json_output:
        arguments.append("--json")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert calls == [capsule_path]
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["ready"] is True
        assert payload["ollama_version"] == "0.33.2"
    else:
        assert "0.33.2" in result.stdout
        assert "qwen3.5:4b" in result.stdout
        assert "nomic-embed-text" in result.stdout


def test_doctor_returns_failure_for_missing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path = tmp_path / "flight.zip"
    capsule_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        cli,
        "check_capsule_readiness",
        lambda path: _report(path, missing=("qwen3.5:4b",)),
    )

    result = CliRunner().invoke(cli.app, ["doctor", str(capsule_path)])

    assert result.exit_code != 0
    assert "qwen3.5:4b" in result.output


def test_doctor_reports_validation_or_connection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path = tmp_path / "flight.zip"
    capsule_path.write_bytes(b"placeholder")

    def fail(*args: Any, **kwargs: Any) -> ReadinessReport:
        request = httpx.Request("GET", "http://127.0.0.1:11434/api/version")
        raise httpx.ConnectError("connection failed", request=request)

    monkeypatch.setattr(cli, "check_capsule_readiness", fail)
    result = CliRunner().invoke(cli.app, ["doctor", str(capsule_path)])

    assert result.exit_code != 0
    assert "connection failed" in result.output
