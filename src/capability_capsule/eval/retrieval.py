"""Score retrieved chunks against expected repository-relative file paths"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.rag.index import SearchResult


class RetrievalMetrics(BaseModel):
    """Metrics for one query within the first k retrieved chunks"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit: bool
    recall: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)


def score_retrieval(
    results: Sequence[SearchResult],
    expected_paths: Sequence[str],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Compare ranked results with expected paths using exact matching"""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not expected_paths:
        raise ValueError("At least one expected path is required")

    if any(not path.strip() for path in expected_paths):
        raise ValueError("Expected paths must not be blank")

    expected = set(expected_paths)
    matched: set[str] = set()
    reciprocal_rank = 0.0

    for rank, result in enumerate(results[:top_k], start=1):
        path = result.chunk.relative_path

        if path in expected:
            matched.add(path)

            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
    return RetrievalMetrics(
        hit=bool(matched),
        recall=len(matched) / len(expected),
        reciprocal_rank=reciprocal_rank,
    )


class RetrievalSummary(BaseModel):
    """Aggregate retrieval metrics with equal weight per query"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int = Field(gt=0)
    hit_rate: float = Field(ge=0, le=1)
    mean_recall: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)


def summarize_retrieval(
    metrics: Sequence[RetrievalMetrics],
) -> RetrievalSummary:
    """Average per-query metrics, including queries with no hits"""

    if not metrics:
        raise ValueError("At least one query result is required")

    count = len(metrics)

    return RetrievalSummary(
        query_count=count,
        hit_rate=sum(int(metric.hit) for metric in metrics) / count,
        mean_recall=sum(metric.recall for metric in metrics) / count,
        mrr=sum(metric.reciprocal_rank for metric in metrics) / count,
    )
