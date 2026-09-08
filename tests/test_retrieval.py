import json
from pathlib import Path

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.retrieval import retrieve
from capability_capsule.rag.storage import save_index


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    chunks = [
        TextChunk(
            relative_path=f"notes/{i}.md",
            source_type=SourceType.REPO,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for i, text in enumerate(["local retrieval", "other topic"])
    ]
    path = tmp_path / "index.npz"
    save_index(
        path,
        chunks,
        [[1.0, 0.0], [0.0, 1.0]],
        embedding_model="nomic-embed-text",
    )
    return path


def _no_request(request: httpx.Request) -> httpx.Response:
    pytest.fail("Invalid input or index must be rejected before an HTTP request")


def test_retrieve_embeds_question_and_returns_matching_chunk(index_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["input"] == ["How does local retrieval work?"]
        assert payload["model"] == "nomic-embed-text"
        return httpx.Response(200, json={"embeddings": [[2.0, 0.0]]})

    before = index_path.read_bytes()
    results = retrieve(
        "How does local retrieval work?",
        index_path,
        Settings(),
        top_k=1,
        transport=httpx.MockTransport(handle),
    )
    assert len(calls) == 1
    assert len(results) == 1
    assert results[0].chunk.text == "local retrieval"
    assert results[0].chunk.relative_path == "notes/0.md"
    assert results[0].score == pytest.approx(1.0)
    assert index_path.read_bytes() == before


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_rejected(index_path: Path, question: str) -> None:
    with pytest.raises(ValueError):
        retrieve(question, index_path, Settings(), transport=httpx.MockTransport(_no_request))


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(index_path: Path, top_k: int) -> None:
    with pytest.raises(ValueError):
        retrieve(
            "question",
            index_path,
            Settings(),
            top_k=top_k,
            transport=httpx.MockTransport(_no_request),
        )


def test_wrong_model_is_rejected_before_embedding(index_path: Path) -> None:
    settings = Settings()
    settings.ollama.embedding_model = "different-model"
    with pytest.raises(ValueError):
        retrieve("question", index_path, settings, transport=httpx.MockTransport(_no_request))


def test_missing_index_is_rejected_before_embedding(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        retrieve(
            "question",
            tmp_path / "missing.npz",
            Settings(),
            transport=httpx.MockTransport(_no_request),
        )


def test_http_failure_is_propagated(index_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        retrieve("question", index_path, Settings(), transport=transport)


def test_query_dimension_mismatch_is_rejected(index_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]})
    )
    with pytest.raises(ValueError):
        retrieve("question", index_path, Settings(), transport=transport)
