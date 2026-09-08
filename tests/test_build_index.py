import json
from pathlib import Path

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.packager.build import build_index
from capability_capsule.rag.storage import load_index


def _no_request(request: httpx.Request) -> httpx.Response:
    pytest.fail("This case must fail before calling Ollama")


def test_build_saves_a_searchable_index_with_ordered_batches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "a.txt").write_text("abcdefgh", encoding="utf-8")
    (repo / "b.md").write_text("ijkl", encoding="utf-8")
    (repo / "ignored.txt").write_text("ignore me", encoding="utf-8")
    (repo / "binary.txt").write_bytes(b"\x00binary")
    (repo / "large.txt").write_text("x" * 40, encoding="utf-8")
    output = tmp_path / "index.npz"
    batches: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "nomic-embed-text"
        batches.append(payload["input"])
        vectors = [[1.0, 0.0] if text == "abcd" else [0.0, 1.0] for text in payload["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    result = build_index(
        repo,
        output,
        Settings(),
        chunk_size_chars=4,
        overlap_chars=0,
        batch_size=2,
        max_file_size_bytes=32,
        transport=httpx.MockTransport(handle),
    )

    assert batches == [["abcd", "efgh"], ["ijkl"]]
    assert result.output_path == output.resolve()
    assert result.document_count == 2
    assert result.chunk_count == 3
    assert result.vector_dimensions == 2
    assert result.size_bytes == output.stat().st_size > 0
    index = load_index(output, expected_embedding_model="nomic-embed-text")
    match = index.search([1.0, 0.0], top_k=1)[0]
    assert match.chunk.relative_path == "a.txt"
    assert match.chunk.text == "abcd"


def test_existing_output_is_preserved_without_http_calls(tmp_path: Path) -> None:
    output = tmp_path / "index.npz"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_index(tmp_path, output, Settings(), transport=httpx.MockTransport(_no_request))
    assert output.read_bytes() == b"existing"


def test_no_chunks_fails_without_output_or_http_calls(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    output = tmp_path / "index.npz"
    with pytest.raises(ValueError):
        build_index(tmp_path, output, Settings(), transport=httpx.MockTransport(_no_request))
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"chunk_size_chars": 0},
        {"overlap_chars": -1},
        {"chunk_size_chars": 5, "overlap_chars": 5},
        {"max_file_size_bytes": 0},
    ],
)
def test_bad_options_fail_before_http_calls(tmp_path: Path, options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        build_index(
            tmp_path,
            tmp_path / "index.npz",
            Settings(),
            transport=httpx.MockTransport(_no_request),
            **options,
        )


@pytest.mark.parametrize("failure", ["http", "dimensions"])
def test_later_batch_failure_does_not_leave_an_index(tmp_path: Path, failure: str) -> None:
    (tmp_path / "input.txt").write_text("abcdefgh", encoding="utf-8")
    output = tmp_path / "index.npz"
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
        if failure == "http":
            return httpx.Response(500, json={"error": "unavailable"})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]})

    expected_error = httpx.HTTPStatusError if failure == "http" else ValueError
    with pytest.raises(expected_error):
        build_index(
            tmp_path,
            output,
            Settings(),
            chunk_size_chars=4,
            overlap_chars=0,
            batch_size=1,
            transport=httpx.MockTransport(handle),
        )
    assert calls == 2
    assert not output.exists()
