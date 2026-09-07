import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import capability_capsule.packager.build as builder
from capability_capsule.config import Settings
from capability_capsule.rag.storage import load_index


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.md").write_text("Local knowledge for offline work.", encoding="utf-8")
    return root


def _transport() -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in texts]})

    return httpx.MockTransport(handle)


def test_build_within_budget_saves_loadable_index(repo: Path, tmp_path: Path) -> None:
    settings = Settings()
    settings.capsule.size_budget_bytes = 1_000_000
    output = tmp_path / "index.npz"
    result = builder.build_index(repo, output, settings, transport=_transport())

    assert 0 < result.size_bytes <= settings.capsule.size_budget_bytes
    assert result.size_bytes == output.stat().st_size
    index = load_index(output, expected_embedding_model=settings.ollama.embedding_model)
    assert index.search([1.0, 0.0])[0].chunk.relative_path == "notes.md"
    assert {path.name for path in tmp_path.iterdir()} == {"repo", "index.npz"}


@pytest.mark.parametrize("budget_delta", [0, -1])
def test_budget_boundary_uses_actual_compressed_size(
    repo: Path, tmp_path: Path, budget_delta: int,
) -> None:
    settings = Settings()
    baseline = builder.build_index(
        repo, tmp_path / "baseline.npz", settings, transport=_transport(),
    )
    settings.capsule.size_budget_bytes = baseline.size_bytes + budget_delta
    output = tmp_path / "bounded.npz"
    before = set(tmp_path.iterdir())

    if budget_delta == 0:
        result = builder.build_index(repo, output, settings, transport=_transport())
        assert result.size_bytes == settings.capsule.size_budget_bytes
        assert set(tmp_path.iterdir()) == before | {output}
    else:
        with pytest.raises(ValueError):
            builder.build_index(repo, output, settings, transport=_transport())
        assert not output.exists()
        assert set(tmp_path.iterdir()) == before


def test_existing_output_is_preserved(repo: Path, tmp_path: Path) -> None:
    output = tmp_path / "index.npz"
    output.write_bytes(b"existing data")
    settings = Settings()
    settings.capsule.size_budget_bytes = 1

    with pytest.raises(FileExistsError):
        builder.build_index(repo, output, settings, transport=_transport())
    assert output.read_bytes() == b"existing data"


def test_save_failure_leaves_no_partial_index(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "index.npz"
    before = set(tmp_path.iterdir())

    def fail(path: Path, *args: Any, **kwargs: Any) -> None:
        path.write_bytes(b"partial archive")
        raise OSError("simulated write failure")

    monkeypatch.setattr(builder, "save_index", fail)
    with pytest.raises(OSError):
        builder.build_index(repo, output, Settings(), transport=_transport())
    assert not output.exists()
    assert set(tmp_path.iterdir()) == before


def test_output_created_during_embedding_is_not_overwritten(repo: Path, tmp_path: Path) -> None:
    output = tmp_path / "index.npz"

    def handle(request: httpx.Request) -> httpx.Response:
        output.write_bytes(b"created by another process")
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    with pytest.raises(FileExistsError):
        builder.build_index(repo, output, Settings(), transport=httpx.MockTransport(handle))
    assert output.read_bytes() == b"created by another process"
    assert {path.name for path in tmp_path.iterdir()} == {"repo", "index.npz"}
