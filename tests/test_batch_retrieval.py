import importlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    chunks = tuple(
        TextChunk(
            relative_path=f"notes/{number}.md", source_type=SourceType.REPO,
            chunk_index=0, start_char=0, end_char=len(text), text=text,
        )
        for number, text in enumerate(("alpha", "beta", "gamma"))
    )
    path = tmp_path / "index.npz"
    save_index(
        path, chunks, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        embedding_model="nomic-embed-text",
    )
    return path


def test_retrieve_many_batches_embedding_and_preserves_question_order(
    index_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("capability_capsule.rag.retrieval")
    original_load = module.load_index
    loads: list[Path] = []

    def counted_load(path: Path, *, expected_embedding_model: str) -> Any:
        loads.append(path)
        return original_load(path, expected_embedding_model=expected_embedding_model)

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert payload["input"] == ["alpha question", "beta question"]
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    monkeypatch.setattr(module, "load_index", counted_load)
    before = index_path.read_bytes()
    batches = module.retrieve_many(
        ("alpha question", "beta question"), index_path, Settings(), top_k=2,
        transport=httpx.MockTransport(handle),
    )
    assert loads == [index_path]
    assert len(requests) == 1
    assert len(batches) == 2
    assert [[row.chunk.relative_path for row in batch] for batch in batches] == [
        ["notes/0.md", "notes/1.md"],
        ["notes/1.md", "notes/0.md"],
    ]
    assert all(len(batch) == 2 for batch in batches)
    assert index_path.read_bytes() == before


def test_retrieve_many_empty_input_does_no_work(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("capability_capsule.rag.retrieval")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Empty input must not load an index or request embeddings")

    monkeypatch.setattr(module, "load_index", forbidden)
    monkeypatch.setattr(module, "embed_texts", forbidden)
    assert module.retrieve_many((), Path("missing.npz"), Settings()) == ()


@pytest.mark.parametrize("questions", [("valid", ""), (" ",), ("valid", "\n\t")])
def test_retrieve_many_rejects_any_blank_question_before_work(
    monkeypatch: pytest.MonkeyPatch, questions: tuple[str, ...],
) -> None:
    module = importlib.import_module("capability_capsule.rag.retrieval")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid questions must be rejected before index loading")

    monkeypatch.setattr(module, "load_index", forbidden)
    with pytest.raises(ValueError):
        module.retrieve_many(questions, Path("unused.npz"), Settings())


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_many_rejects_invalid_top_k_before_work(
    monkeypatch: pytest.MonkeyPatch, top_k: int,
) -> None:
    module = importlib.import_module("capability_capsule.rag.retrieval")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid top_k must be rejected before index loading")

    monkeypatch.setattr(module, "load_index", forbidden)
    with pytest.raises(ValueError):
        module.retrieve_many(("question",), Path("unused.npz"), Settings(), top_k=top_k)
