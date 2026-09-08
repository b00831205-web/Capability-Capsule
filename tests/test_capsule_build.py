import json
import zipfile
from pathlib import Path

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import CreatedBy, SourceType
from capability_capsule.rag.storage import load_index


def test_build_capsule_creates_index_manifest_and_provenance(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["build_capsule"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("abcdefghij", encoding="utf-8")
    (repo / "b.md").write_text("klmn", encoding="utf-8")
    output = tmp_path / "flight capsule.zip"
    settings = Settings()
    requests: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        texts: list[str] = payload["input"]
        requests.append(texts)
        return httpx.Response(
            200,
            json={
                "embeddings": [[float(len(text)), 1.0] for text in texts],
            },
        )

    result = module.build_capsule(
        repo,
        output,
        "continue offline development",
        settings,
        chunk_size_chars=4,
        overlap_chars=0,
        batch_size=2,
        transport=httpx.MockTransport(handle),
    )
    assert requests == [["abcd", "efgh"], ["ij", "klmn"]]
    assert result.output_path == output.resolve()
    assert result.document_count == 2
    assert result.chunk_count == 4
    assert result.vector_dimensions == 2
    assert result.size_bytes == output.stat().st_size
    assert result.manifest.task == "continue offline development"
    assert result.manifest.size_budget_bytes == settings.capsule.size_budget_bytes
    assert result.manifest.offline_duration_hours == settings.capsule.offline_duration_hours
    assert result.manifest.generation_model == settings.ollama.generation_model
    assert result.manifest.embedding_model == settings.ollama.embedding_model
    assert [artifact.source for artifact in result.manifest.artifacts] == ["a.md", "b.md"]
    assert all(
        artifact.source_type is SourceType.REPO
        and artifact.created_by is CreatedBy.ORIGINAL
        and artifact.capsule_build_id == result.manifest.capsule_build_id
        for artifact in result.manifest.artifacts
    )

    inspection = module.inspect_capsule(output)
    assert inspection.manifest == result.manifest
    assert inspection.package_size_bytes == result.size_bytes
    with zipfile.ZipFile(output) as archive:
        extracted = tmp_path / "extracted-index.npz"
        extracted.write_bytes(archive.read("index.npz"))
    index = load_index(extracted, expected_embedding_model=settings.ollama.embedding_model)
    assert [(chunk.source_type, chunk.relative_path) for chunk in index.chunks] == [
        (SourceType.REPO, "a.md"),
        (SourceType.REPO, "a.md"),
        (SourceType.REPO, "a.md"),
        (SourceType.REPO, "b.md"),
    ]
    assert not list(tmp_path.glob(".capsule-build-*"))


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_build_capsule_rejects_blank_task_before_embedding(tmp_path: Path, task: str) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["build_capsule"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("text", encoding="utf-8")

    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("Blank task must be rejected before embedding")

    output = tmp_path / "capsule.zip"
    with pytest.raises(ValueError):
        module.build_capsule(
            repo,
            output,
            task,
            Settings(),
            transport=httpx.MockTransport(forbidden),
        )
    assert not output.exists()


def test_build_capsule_refuses_existing_output_before_embedding(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["build_capsule"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("text", encoding="utf-8")
    output = tmp_path / "capsule.zip"
    output.write_bytes(b"existing")

    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("Existing output must be rejected before embedding")

    with pytest.raises(FileExistsError):
        module.build_capsule(
            repo,
            output,
            "task",
            Settings(),
            transport=httpx.MockTransport(forbidden),
        )
    assert output.read_bytes() == b"existing"


def test_build_capsule_cleans_up_after_embedding_failure(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["build_capsule"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("text", encoding="utf-8")
    output = tmp_path / "capsule.zip"
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        module.build_capsule(repo, output, "task", Settings(), transport=transport)
    assert not output.exists()
    assert not list(tmp_path.glob(".capsule-build-*"))


def test_build_capsule_rejects_missing_output_parent_before_embedding(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["build_capsule"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.md").write_text("text", encoding="utf-8")

    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("Missing parent must be rejected before embedding")

    output = tmp_path / "missing" / "capsule.zip"
    with pytest.raises(NotADirectoryError):
        module.build_capsule(
            repo,
            output,
            "task",
            Settings(),
            transport=httpx.MockTransport(forbidden),
        )
    assert not output.exists()
