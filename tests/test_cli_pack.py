import json
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.config import Settings
from capability_capsule.manifest import CapsuleManifest
from capability_capsule.packager.capsule import CapsuleBuildResult


def _result(output: Path, settings: Settings) -> CapsuleBuildResult:
    manifest = CapsuleManifest(
        task="continue offline development",
        size_budget_bytes=settings.capsule.size_budget_bytes,
        offline_duration_hours=settings.capsule.offline_duration_hours,
        generation_model=settings.ollama.generation_model,
        embedding_model=settings.ollama.embedding_model,
    )
    return CapsuleBuildResult(
        output_path=output.resolve(),
        manifest=manifest,
        document_count=2,
        chunk_count=4,
        vector_dimensions=768,
        size_bytes=12_345,
    )


@pytest.mark.parametrize("as_json", [False, True])
def test_pack_cli_builds_capsule_and_outputs_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_json: bool,
) -> None:
    output = tmp_path / "flight capsule.zip"
    config = tmp_path / "config.toml"
    config.write_text(
        '[ollama]\ngeneration_model = "test-generation"\n'
        "[capsule]\nsize_budget_bytes = 2000000\noffline_duration_hours = 10\n",
        encoding="utf-8",
    )
    expected_settings = Settings.from_toml(config)
    expected = _result(output, expected_settings)
    calls: list[bool] = []

    def fake_build(
        root: Path,
        output_path: Path,
        task: str,
        settings: Settings,
    ) -> CapsuleBuildResult:
        assert root.resolve() == tmp_path.resolve()
        assert output_path == output
        assert task == "continue offline development"
        assert settings.model_dump(mode="json") == expected_settings.model_dump(mode="json")
        calls.append(True)
        return expected

    monkeypatch.setattr(cli, "build_capsule", fake_build, raising=False)
    args = [
        "pack",
        "--repo",
        str(tmp_path),
        "--output",
        str(output),
        "--task",
        "continue offline development",
        "--config",
        str(config),
    ]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert not result.stderr.strip()
    if as_json:
        assert json.loads(result.stdout) == expected.model_dump(mode="json")
    else:
        assert result.stdout.strip()
        assert str(expected.manifest.capsule_build_id) in result.stdout
        assert "2" in result.stdout
        assert "4" in result.stdout
        assert "768" in result.stdout
        assert "12345" in result.stdout.replace(",", "")


@pytest.mark.parametrize(
    "failure",
    [FileExistsError("exists"), ValueError("invalid"), httpx.ConnectError("offline")],
)
def test_pack_failure_has_no_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    calls: list[bool] = []

    def fail(*args: Any, **kwargs: Any) -> CapsuleBuildResult:
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "build_capsule", fail, raising=False)
    result = CliRunner().invoke(
        cli.app,
        [
            "pack",
            "--repo",
            str(tmp_path),
            "--output",
            str(tmp_path / "capsule.zip"),
            "--task",
            "task",
            "--json",
        ],
    )
    assert calls == [True]
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not result.stdout.strip()
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("missing", ["task", "repo", "output"])
def test_pack_requires_core_inputs(tmp_path: Path, missing: str) -> None:
    args = [
        "pack",
        "--repo",
        str(tmp_path),
        "--output",
        str(tmp_path / "capsule.zip"),
        "--task",
        "task",
    ]
    option = {"repo": "--repo", "output": "--output", "task": "--task"}[missing]
    position = args.index(option)
    del args[position : position + 2]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)


def test_pack_json_serializes_build_id_as_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _result(tmp_path / "capsule.zip", Settings())
    monkeypatch.setattr(cli, "build_capsule", lambda *args, **kwargs: expected, raising=False)
    result = CliRunner().invoke(
        cli.app,
        [
            "pack",
            "--repo",
            str(tmp_path),
            "--output",
            str(tmp_path / "capsule.zip"),
            "--task",
            "task",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert UUID(json.loads(result.stdout)["manifest"]["capsule_build_id"])
