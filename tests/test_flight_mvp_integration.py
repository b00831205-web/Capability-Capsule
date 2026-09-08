import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from capability_capsule.cli import app

pytestmark = pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_OLLAMA") != "1",
    reason="Set CAPSULE_RUN_OLLAMA=1 to use local Ollama",
)


def test_real_pack_doctor_run_report_flow(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "flight.md").write_text(
        "Flight mode uses a portable local vector index and local Ollama models.",
        encoding="utf-8",
    )
    capsule_path = tmp_path / "flight.zip"
    runner = CliRunner()

    packed = runner.invoke(
        app,
        [
            "pack",
            "--repo",
            str(repository),
            "--output",
            str(capsule_path),
            "--task",
            "answer questions during a flight",
            "--json",
        ],
    )
    assert packed.exit_code == 0, packed.output
    assert capsule_path.is_file()

    checked = runner.invoke(app, ["doctor", str(capsule_path), "--json"])
    assert checked.exit_code == 0, checked.output
    readiness = json.loads(checked.stdout)
    assert readiness["ready"] is True
    assert readiness["missing_models"] == []

    answered = runner.invoke(
        app,
        [
            "run",
            str(capsule_path),
            "What does flight mode use?",
            "--top-k",
            "1",
            "--json",
        ],
    )
    assert answered.exit_code == 0, answered.output
    answer = json.loads(answered.stdout)
    assert answer["answer"].strip()
    assert answer["sources"][0]["chunk"]["relative_path"] == "flight.md"

    telemetry_dir = tmp_path / ".capsule" / "sessions"
    reported = runner.invoke(app, ["report", str(telemetry_dir), "--json"])
    assert reported.exit_code == 0, reported.output
    report = json.loads(reported.stdout)
    assert report["event_count"] == 1
    assert report["success_count"] == 1
    assert report["error_count"] == 0
