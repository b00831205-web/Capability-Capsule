"""Run retrieval evaluation against a saved index."""

from collections.abc import Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from capability_capsule.config import Settings
from capability_capsule.eval.retrieval import (
    RetrievalMetrics,
    RetrievalSummary,
    score_retrieval,
    summarize_retrieval,
)
from capability_capsule.rag.index import SearchResult
from capability_capsule.rag.retrieval import retrieve_many


class RetrievalCase(BaseModel):
    """One question and its expected repository-relative source path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    expected_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value

    @field_validator("expected_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not path.strip() for path in value):
            raise ValueError("Expected paths must not be blank")

        return value


class RetrievalCaseResult(BaseModel):
    """Per-case results and aggregate metrics for a completed run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case: RetrievalCase
    sources: tuple[SearchResult, ...]
    metrics: RetrievalMetrics


class RetrievalEvaluationReport(BaseModel):
    """Per-case results and aggregate metrics for a completed run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(gt=0)
    cases: tuple[RetrievalCaseResult, ...] = Field(min_length=1)
    summary: RetrievalSummary


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    index_path: Path,
    settings: Settings,
    *,
    top_k: int = 5,
    transport: httpx.BaseTransport | None = None,
) -> RetrievalEvaluationReport:
    """Evaluate each case without generation answers or modifying the index."""

    if not cases:
        raise ValueError("At least one evaluation case is required")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results: list[RetrievalCaseResult] = []
    case_batch = tuple(cases)

    source_batches = retrieve_many(
        tuple(case.question for case in case_batch),
        index_path,
        settings,
        top_k=top_k,
        transport=transport,
    )

    for case, sources in zip(
        case_batch,
        source_batches,
        strict=True,
    ):
        metrics = score_retrieval(sources, case.expected_paths, top_k=top_k)
        results.append(RetrievalCaseResult(case=case, sources=sources, metrics=metrics))
    return RetrievalEvaluationReport(
        top_k=top_k,
        cases=tuple(results),
        summary=summarize_retrieval(tuple(result.metrics for result in results)),
    )
