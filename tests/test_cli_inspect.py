import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.config import Settings
from capability_capsule.manifest import CapsuleManifest
from capability_capsule.packager.capsule import CapsuleInspection


def _inspection(path: Path) -> CapsuleInspection:
    settings = Settings()
    manifest = CapsuleManifest(
        task="continue offline development",
        size_budget_bytes=settings.capsule.size_budget_bytes,
        offline_duration_hours=settings.capsule.offline_duration_hours,
        generation_model=settings.ollama.generation_model,
        embedding_model=settings.ollama.embedding_model,
    )
    return CapsuleInspection(
        path=path.resolve(),
        manifest=manifest,
        settings=settings,
        index_size_bytes=10_000,
        package_size_bytes=12_345,
    )


@pytest.mark.parametrize("as_json", [False, True])
def test_inspect_cli_outputs_validated_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_json: bool,
) -> None:
    capsule = tmp_path / "flight capsule.zip"
    capsule.touch()
    expected = _inspection(capsule)
    calls: list[Path] = []

    def fake_inspect(path: Path) -> CapsuleInspection:
        calls.append(path)
        return expected

    monkeypatch.setattr(cli, "inspect_capsule", fake_inspect, raising=False)
    args = ["inspect", str(capsule)]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [capsule]
    assert not result.stderr.strip()
    if as_json:
        assert json.loads(result.stdout) == expected.model_dump(mode="json")
    else:
        assert result.stdout.strip()
        assert expected.manifest.task in result.stdout
        assert str(expected.manifest.capsule_build_id) in result.stdout
        assert expected.manifest.generation_model in result.stdout
        assert expected.manifest.embedding_model in result.stdout
        assert "10000" in result.stdout.replace(",", "")
        assert "12345" in result.stdout.replace(",", "")


@pytest.mark.parametrize("failure", [ValueError("invalid"), OSError("unreadable")])
def test_inspect_cli_failure_has_no_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    capsule = tmp_path / "capsule.zip"
    capsule.touch()
    calls: list[bool] = []

    def fail(*args: Any, **kwargs: Any) -> CapsuleInspection:
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "inspect_capsule", fail, raising=False)
    result = CliRunner().invoke(cli.app, ["inspect", str(capsule), "--json"])
    assert calls == [True]
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not result.stdout.strip()
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


def test_inspect_cli_requires_existing_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["inspect", str(tmp_path / "missing.zip")])
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
