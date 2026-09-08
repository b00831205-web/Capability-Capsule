import importlib
from typing import Any

import pytest

from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult


def _result(path: str, number: int) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            relative_path=path,
            source_type=SourceType.REPO,
            chunk_index=number,
            start_char=0,
            end_char=4,
            text="text",
        ),
        score=1.0 / (number + 1),
    )


@pytest.mark.parametrize(
    ("paths", "expected", "top_k", "hit", "recall", "reciprocal_rank"),
    [
        (["a.md"], ["a.md"], 1, True, 1.0, 1.0),
        (["x.md", "a.md"], ["a.md"], 2, True, 1.0, 0.5),
        (["x.md", "a.md"], ["a.md"], 1, False, 0.0, 0.0),
        ([], ["a.md"], 5, False, 0.0, 0.0),
        (["a.md", "a.md", "b.md"], ["a.md", "b.md"], 2, True, 0.5, 1.0),
        (["a.md"], ["a.md", "a.md", "b.md"], 5, True, 0.5, 1.0),
        (["x.md", "x.md", "a.md"], ["a.md"], 3, True, 1.0, 1 / 3),
        (["docs/a.md"], ["a.md"], 5, False, 0.0, 0.0),
    ],
)
def test_retrieval_metrics(
    paths: list[str],
    expected: list[str],
    top_k: int,
    hit: bool,
    recall: float,
    reciprocal_rank: float,
) -> None:
    module = importlib.import_module("capability_capsule.eval.retrieval")
    results = tuple(_result(path, number) for number, path in enumerate(paths))
    before = tuple(result.model_dump() for result in results)
    metrics = module.score_retrieval(results, expected, top_k=top_k)
    assert metrics.hit is hit
    assert metrics.recall == pytest.approx(recall)
    assert metrics.reciprocal_rank == pytest.approx(reciprocal_rank)
    assert tuple(result.model_dump() for result in results) == before


@pytest.mark.parametrize(
    ("expected", "top_k"),
    [([], 5), ([""], 5), (["   "], 5), (["a.md"], 0), (["a.md"], -1)],
)
def test_retrieval_metrics_reject_invalid_inputs(expected: list[str], top_k: int) -> None:
    module: Any = importlib.import_module("capability_capsule.eval.retrieval")
    with pytest.raises(ValueError):
        module.score_retrieval((), expected, top_k=top_k)
