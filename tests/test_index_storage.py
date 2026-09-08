import json
from pathlib import Path

import numpy as np
import pytest

from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import VectorIndex
from capability_capsule.rag.storage import load_index, save_index


def _chunks() -> list[TextChunk]:
    return [
        TextChunk(
            relative_path=f"notes/{i}.md",
            source_type=SourceType.REPO,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for i, text in enumerate(["离线检索\n", "local model"])
    ]


def test_round_trip_preserves_chunks_and_search_results(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    chunks = _chunks()
    vectors = [[2.0, 0.0], [1.0, 1.0]]
    original = VectorIndex(chunks, vectors)

    save_index(path, chunks, vectors, embedding_model="nomic-embed-text")
    restored = load_index(path, expected_embedding_model="nomic-embed-text")

    expected = original.search([1.0, 0.0])
    actual = restored.search([1.0, 0.0])
    assert [result.chunk for result in actual] == [result.chunk for result in expected]
    assert [result.score for result in actual] == pytest.approx(
        [result.score for result in expected]
    )


def test_save_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    path.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        save_index(path, _chunks(), [[1.0, 0.0], [0.0, 1.0]], embedding_model="model")
    assert path.read_bytes() == b"existing data"


def test_invalid_vectors_are_rejected_before_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    with pytest.raises(ValueError):
        save_index(path, _chunks(), [[0.0, 0.0], [1.0, 0.0]], embedding_model="model")
    assert not path.exists()


def test_load_rejects_different_embedding_model(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    save_index(path, _chunks(), [[1.0, 0.0], [0.0, 1.0]], embedding_model="model-a")
    with pytest.raises(ValueError):
        load_index(path, expected_embedding_model="model-b")


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "missing.npz", expected_embedding_model="model")


def test_load_rejects_non_archive(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not an index")
    with pytest.raises(ValueError):
        load_index(path, expected_embedding_model="model")


@pytest.mark.parametrize("invalid_part", ["version", "count", "zero", "metadata"])
def test_load_validates_archive_contents(tmp_path: Path, invalid_part: str) -> None:
    path = tmp_path / "bad.npz"
    metadata = {
        "schema_version": "99" if invalid_part == "version" else "0.1",
        "embedding_model": "model",
        "chunks": [chunk.model_dump(mode="json") for chunk in _chunks()],
    }
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    if invalid_part == "count":
        vectors = [[1.0, 0.0]]
    elif invalid_part == "zero":
        vectors[0] = [0.0, 0.0]
    metadata_text = "not json" if invalid_part == "metadata" else json.dumps(metadata)
    np.savez_compressed(path, metadata=np.array(metadata_text), vectors=np.array(vectors))

    with pytest.raises(ValueError):
        load_index(path, expected_embedding_model="model")
