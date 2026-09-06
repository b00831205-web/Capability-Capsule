import pytest

from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import VectorIndex


def _chunk(name: str) -> TextChunk:
    return TextChunk(
        relative_path=f"notes/{name}.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=len(name),
        text=name,
    )


def test_search_uses_cosine_similarity_and_preserves_chunk_metadata() -> None:
    chunks = [_chunk("diagonal"), _chunk("aligned"), _chunk("opposite")]
    index = VectorIndex(chunks, [[100.0, 100.0], [2.0, 0.0], [-1.0, 0.0]])

    results = index.search([3.0, 0.0], top_k=2)

    assert [result.chunk for result in results] == [chunks[1], chunks[0]]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)
    all_results = index.search([1.0, 0.0], top_k=10)
    assert len(all_results) == 3
    assert all_results[-1].score == pytest.approx(-1.0)


def test_ties_keep_input_order() -> None:
    chunks = [_chunk("first"), _chunk("second")]
    index = VectorIndex(chunks, [[1.0, 0.0], [2.0, 0.0]])

    assert [result.chunk for result in index.search([1.0, 0.0])] == chunks


def test_index_is_independent_of_mutated_input_lists() -> None:
    original = _chunk("original")
    chunks = [original]
    vectors = [[1.0, 0.0]]
    index = VectorIndex(chunks, vectors)
    chunks[0] = _chunk("replacement")
    vectors[0][:] = [0.0, 1.0]

    result = index.search([1.0, 0.0])[0]
    assert result.chunk == original
    assert result.score == pytest.approx(1.0)


def test_empty_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        VectorIndex([], [])


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0]],
        [[1.0], [1.0, 2.0]],
        [[], []],
        [[0.0, 0.0], [1.0, 0.0]],
        [[float("nan"), 0.0], [1.0, 0.0]],
        [[float("inf"), 0.0], [1.0, 0.0]],
    ],
)
def test_invalid_index_vectors_are_rejected(vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        VectorIndex([_chunk("first"), _chunk("second")], vectors)


@pytest.mark.parametrize(
    "query",
    [[], [1.0], [1.0, 0.0, 0.0], [0.0, 0.0], [float("nan"), 0.0],
     [float("inf"), 0.0]],
)
def test_invalid_query_is_rejected(query: list[float]) -> None:
    index = VectorIndex([_chunk("first")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        index.search(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    index = VectorIndex([_chunk("first")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], top_k=top_k)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_finite_extreme_magnitudes_can_be_normalized(scale: float) -> None:
    index = VectorIndex([_chunk("first")], [[scale, scale]])
    assert index.search([scale, scale])[0].score == pytest.approx(1.0)
