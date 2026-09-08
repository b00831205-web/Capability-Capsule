import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index
from capability_capsule.runtime.ollama import answer_question


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    text = "Capsule stores a local vector index."
    chunk = TextChunk(
        relative_path="README.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )
    path = tmp_path / "index.npz"
    save_index(path, [chunk], [[1.0, 0.0]], embedding_model="nomic-embed-text")
    return path


def test_answer_retrieves_then_generates_with_numbered_sources(index_path: Path) -> None:
    routes: list[str] = []
    settings = Settings()
    settings.ollama.generation_model = "test-generator"

    def handle(request: httpx.Request) -> httpx.Response:
        routes.append(request.url.path)
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            assert payload["input"] == ["What does Capsule store?"]
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        assert request.url.path == "/api/chat"
        assert payload["model"] == "test-generator"
        assert payload["stream"] is False
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        user_data = json.loads(payload["messages"][1]["content"])
        assert user_data["question"] == "What does Capsule store?"
        source = user_data["sources"][0]
        assert source["id"] == 1
        assert source["path"] == "README.md"
        assert source["text"] == "Capsule stores a local vector index."
        assert source["start_char"] == 0
        assert source["end_char"] == len(source["text"])
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "A local index [1]."},
            },
        )

    result = answer_question(
        "What does Capsule store?",
        index_path,
        settings,
        transport=httpx.MockTransport(handle),
    )
    assert routes == ["/api/embed", "/api/chat"]
    assert result.answer == "A local index [1]."
    assert result.generation_model == "test-generator"
    assert result.sources[0].chunk.relative_path == "README.md"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"done": False},
        {"done": True},
        {"done": True, "message": {"role": "assistant", "content": ""}},
        {"done": True, "message": {"role": "assistant", "content": "  "}},
        {"done": True, "message": {"role": "assistant", "content": 123}},
        {"done": True, "message": {"role": "user", "content": "wrong role"}},
    ],
)
def test_invalid_generation_response_is_rejected(
    index_path: Path,
    payload: dict[str, Any],
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        return httpx.Response(200, json=payload)

    with pytest.raises(ValueError):
        answer_question("question", index_path, Settings(), transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("failure_route", ["/api/embed", "/api/chat"])
def test_http_failure_is_propagated(index_path: Path, failure_route: str) -> None:
    routes: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        routes.append(request.url.path)
        if request.url.path == failure_route:
            return httpx.Response(500)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    with pytest.raises(httpx.HTTPStatusError):
        answer_question("question", index_path, Settings(), transport=httpx.MockTransport(handle))
    assert routes[-1] == failure_route


def test_proxy_bypass_applies_to_both_clients(
    index_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_client = httpx.Client
    inherited: list[bool] = []

    def make_client(**kwargs: Any) -> httpx.Client:
        inherited.append(kwargs["trust_env"])
        return original_client(**kwargs)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "answer"},
            },
        )

    monkeypatch.setattr(httpx, "Client", make_client)
    answer_question("question", index_path, Settings(), transport=httpx.MockTransport(handle))
    assert inherited == [False, False]
