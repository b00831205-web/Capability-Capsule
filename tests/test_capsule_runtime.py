import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import CapsuleManifest, SourceType
from capability_capsule.packager.capsule import write_capsule
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index


def _capsule(tmp_path: Path) -> Path:
    settings = Settings.model_validate(
        {
            "ollama": {
                "generation_model": "capsule-generator",
                "embedding_model": "capsule-embedder",
            }
        }
    )
    index_path = tmp_path / "index.npz"
    text = "The flight capsule carries an offline repository index."
    chunk = TextChunk(
        relative_path="docs/flight.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )
    save_index(index_path, (chunk,), ((1.0, 0.0),), embedding_model="capsule-embedder")
    manifest = CapsuleManifest(
        capsule_build_id=uuid4(),
        task="answer questions during a flight",
        size_budget_bytes=settings.capsule.size_budget_bytes,
        offline_duration_hours=settings.capsule.offline_duration_hours,
        generation_model="capsule-generator",
        embedding_model="capsule-embedder",
    )
    capsule_path = tmp_path / "flight.zip"
    write_capsule(index_path, capsule_path, manifest, settings)
    return capsule_path


def test_answer_from_capsule_uses_embedded_index_and_settings(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.capsule", fromlist=["answer_from_capsule"])
    capsule_path = _capsule(tmp_path)
    before = capsule_path.read_bytes()
    requests: list[tuple[str, dict[str, object]]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "It carries an index [1]."},
            },
        )

    result = module.answer_from_capsule(
        capsule_path,
        "What does the flight capsule carry?",
        transport=httpx.MockTransport(handle),
    )

    assert [path for path, _ in requests] == ["/api/embed", "/api/chat"]
    assert requests[0][1]["model"] == "capsule-embedder"
    assert requests[1][1]["model"] == "capsule-generator"
    assert result.answer == "It carries an index [1]."
    assert result.generation_model == "capsule-generator"
    assert result.sources[0].chunk.relative_path == "docs/flight.md"
    assert capsule_path.read_bytes() == before
    assert not list(tmp_path.glob(".capsule-run-*"))
    events = list((tmp_path / ".capsule" / "sessions").glob("*.json"))
    assert len(events) == 1
    event = json.loads(events[0].read_text(encoding="utf-8"))
    assert event["status"] == "success"
    assert event["source_count"] == 1
    assert event["question_chars"] == len("What does the flight capsule carry?")
    assert "What does the flight capsule carry?" not in events[0].read_text(encoding="utf-8")


def test_answer_from_capsule_forwards_runtime_limits(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.capsule", fromlist=["answer_from_capsule"])
    capsule_path = _capsule(tmp_path)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        payload = json.loads(request.content)
        user_data = json.loads(payload["messages"][1]["content"])
        assert len(user_data["sources"]) == 1
        assert len(user_data["sources"][0]["text"]) == 8
        return httpx.Response(
            200,
            json={"done": True, "message": {"role": "assistant", "content": "answer"}},
        )

    module.answer_from_capsule(
        capsule_path,
        "question",
        top_k=1,
        max_context_chars=8,
        transport=httpx.MockTransport(handle),
    )


def test_answer_from_capsule_records_runtime_failure(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.capsule", fromlist=["answer_from_capsule"])
    capsule_path = _capsule(tmp_path)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        module.answer_from_capsule(
            capsule_path,
            "private question",
            transport=httpx.MockTransport(handle),
        )

    events = list((tmp_path / ".capsule" / "sessions").glob("*.json"))
    assert len(events) == 1
    event = json.loads(events[0].read_text(encoding="utf-8"))
    assert event["status"] == "error"
    assert event["error_type"] == "HTTPStatusError"
    assert event["source_count"] == 0
    assert "private question" not in events[0].read_text(encoding="utf-8")


def test_telemetry_write_failure_does_not_block_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__("capability_capsule.runtime.capsule", fromlist=["answer_from_capsule"])
    capsule_path = _capsule(tmp_path)

    def fail_telemetry(*args: object, **kwargs: object) -> None:
        raise OSError("telemetry disk unavailable")

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        return httpx.Response(
            200,
            json={"done": True, "message": {"role": "assistant", "content": "answer"}},
        )

    monkeypatch.setattr(module, "write_run_telemetry", fail_telemetry)
    result = module.answer_from_capsule(
        capsule_path,
        "question",
        transport=httpx.MockTransport(handle),
    )

    assert result.answer == "answer"
