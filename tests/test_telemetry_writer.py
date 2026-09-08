import json
from pathlib import Path
from uuid import uuid4

from capability_capsule.config import TelemetryConfig


def test_write_run_telemetry_creates_private_local_record(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.writer", fromlist=["write_run_telemetry"])
    capsule_path = tmp_path / "flight.zip"
    build_id = uuid4()
    config = TelemetryConfig(enabled=True, output_dir=Path("telemetry/sessions"))

    output = module.write_run_telemetry(
        config,
        capsule_path=capsule_path,
        capsule_build_id=build_id,
        generation_model="qwen3.5:4b",
        duration_ms=125.5,
        question_chars=42,
        source_count=3,
        status="success",
    )

    assert output is not None
    assert output.parent == tmp_path / "telemetry" / "sessions"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["capsule_build_id"] == str(build_id)
    assert payload["generation_model"] == "qwen3.5:4b"
    assert payload["duration_ms"] == 125.5
    assert payload["question_chars"] == 42
    assert payload["source_count"] == 3
    assert payload["status"] == "success"
    assert payload["error_type"] is None
    serialized = output.read_text(encoding="utf-8")
    assert "secret question" not in serialized
    assert "answer text" not in serialized


def test_write_run_telemetry_can_be_disabled(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.writer", fromlist=["write_run_telemetry"])
    config = TelemetryConfig(enabled=False, output_dir=Path("telemetry"))

    output = module.write_run_telemetry(
        config,
        capsule_path=tmp_path / "flight.zip",
        capsule_build_id=uuid4(),
        generation_model="qwen3.5:4b",
        duration_ms=1.0,
        question_chars=8,
        source_count=0,
        status="success",
    )

    assert output is None
    assert not (tmp_path / "telemetry").exists()


def test_write_run_telemetry_records_error_type(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.writer", fromlist=["write_run_telemetry"])
    config = TelemetryConfig(enabled=True, output_dir=tmp_path / "events")

    output = module.write_run_telemetry(
        config,
        capsule_path=tmp_path / "flight.zip",
        capsule_build_id=uuid4(),
        generation_model="qwen3.5:4b",
        duration_ms=25.0,
        question_chars=12,
        source_count=0,
        status="error",
        error_type="ConnectError",
    )

    assert output is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "ConnectError"


def test_write_run_telemetry_uses_distinct_files(tmp_path: Path) -> None:
    module = __import__("capability_capsule.telemetry.writer", fromlist=["write_run_telemetry"])
    config = TelemetryConfig(enabled=True, output_dir=tmp_path / "events")
    arguments = {
        "capsule_path": tmp_path / "flight.zip",
        "capsule_build_id": uuid4(),
        "generation_model": "qwen3.5:4b",
        "duration_ms": 10.0,
        "question_chars": 5,
        "source_count": 1,
        "status": "success",
    }

    first = module.write_run_telemetry(config, **arguments)
    second = module.write_run_telemetry(config, **arguments)

    assert first is not None
    assert second is not None
    assert first != second
    assert len(list((tmp_path / "events").glob("*.json"))) == 2
