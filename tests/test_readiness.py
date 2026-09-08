from pathlib import Path
from uuid import uuid4

import httpx

from capability_capsule.config import Settings
from capability_capsule.manifest import CapsuleManifest, SourceType
from capability_capsule.packager.capsule import write_capsule
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index


def _capsule(tmp_path: Path) -> Path:
    settings = Settings.model_validate(
        {
            "ollama": {
                "generation_model": "qwen3.5:4b",
                "embedding_model": "nomic-embed-text",
            }
        }
    )
    index_path = tmp_path / "index.npz"
    text = "offline readiness"
    chunk = TextChunk(
        relative_path="README.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )
    save_index(index_path, (chunk,), ((1.0, 0.0),), embedding_model="nomic-embed-text")
    manifest = CapsuleManifest(
        capsule_build_id=uuid4(),
        task="work offline",
        size_budget_bytes=settings.capsule.size_budget_bytes,
        offline_duration_hours=settings.capsule.offline_duration_hours,
        generation_model=settings.ollama.generation_model,
        embedding_model=settings.ollama.embedding_model,
    )
    capsule_path = tmp_path / "flight.zip"
    write_capsule(index_path, capsule_path, manifest, settings)
    return capsule_path


def test_readiness_reports_available_required_models(tmp_path: Path) -> None:
    module = __import__(
        "capability_capsule.runtime.readiness", fromlist=["check_capsule_readiness"]
    )
    capsule_path = _capsule(tmp_path)
    routes: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        routes.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.33.2"})
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3.5:4b"},
                    {"model": "nomic-embed-text:latest"},
                    {"name": "unrelated:latest"},
                ]
            },
        )

    result = module.check_capsule_readiness(
        capsule_path,
        transport=httpx.MockTransport(handle),
    )

    assert routes == ["/api/version", "/api/tags"]
    assert result.ready is True
    assert result.ollama_version == "0.33.2"
    assert result.required_models == ("qwen3.5:4b", "nomic-embed-text")
    assert result.missing_models == ()
    assert result.capsule_build_id is not None


def test_readiness_reports_missing_models_without_downloading(tmp_path: Path) -> None:
    module = __import__(
        "capability_capsule.runtime.readiness", fromlist=["check_capsule_readiness"]
    )
    capsule_path = _capsule(tmp_path)
    routes: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        routes.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.33.2"})
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})

    result = module.check_capsule_readiness(
        capsule_path,
        transport=httpx.MockTransport(handle),
    )

    assert result.ready is False
    assert set(result.missing_models) == {"qwen3.5:4b", "nomic-embed-text"}
    assert routes == ["/api/version", "/api/tags"]


def test_readiness_accepts_latest_alias_for_untagged_requirement(tmp_path: Path) -> None:
    module = __import__(
        "capability_capsule.runtime.readiness", fromlist=["check_capsule_readiness"]
    )
    capsule_path = _capsule(tmp_path)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.33.2"})
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3.5:4b"},
                    {"name": "nomic-embed-text:latest"},
                ]
            },
        )

    result = module.check_capsule_readiness(
        capsule_path,
        transport=httpx.MockTransport(handle),
    )

    assert result.ready is True


def test_readiness_rejects_invalid_ollama_payloads(tmp_path: Path) -> None:
    module = __import__(
        "capability_capsule.runtime.readiness", fromlist=["check_capsule_readiness"]
    )
    capsule_path = _capsule(tmp_path)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"models": []})

    try:
        module.check_capsule_readiness(
            capsule_path,
            transport=httpx.MockTransport(handle),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Ollama metadata should be rejected")
