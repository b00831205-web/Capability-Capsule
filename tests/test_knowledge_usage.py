from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.config import TelemetryConfig
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult
from capability_capsule.telemetry.knowledge_usage import (
    KnowledgeNodeReference,
    KnowledgeUsageEvent,
    build_retrieval_usage_event,
    knowledge_node_id,
    knowledge_node_reference_id,
    summarize_knowledge_usage,
    summarize_knowledge_usage_directory,
    write_knowledge_usage_event,
)


def event(
    *,
    required_node_ids: tuple[str, ...],
    activated_node_ids: tuple[str, ...],
    cache_eligible: bool,
    cache_hit: bool,
    cold_start: bool,
    latency_saved_ms: float = 0.0,
    input_tokens_saved: int = 0,
) -> KnowledgeUsageEvent:
    return KnowledgeUsageEvent(
        capsule_id="capsule-smoke-001",
        knowledge_tree_digest="a" * 64,
        request_id="request-001",
        task_family_id="cli-search",
        required_node_ids=required_node_ids,
        activated_node_ids=activated_node_ids,
        cache_eligible=cache_eligible,
        cache_hit=cache_hit,
        cold_start=cold_start,
        latency_saved_ms=latency_saved_ms,
        input_tokens_saved=input_tokens_saved,
    )


def test_summary_separates_coverage_from_cache_reuse() -> None:
    summary = summarize_knowledge_usage(
        [
            event(
                required_node_ids=("cli", "search"),
                activated_node_ids=("cli", "search"),
                cache_eligible=False,
                cache_hit=False,
                cold_start=True,
            ),
            event(
                required_node_ids=("cli", "search"),
                activated_node_ids=("cli",),
                cache_eligible=True,
                cache_hit=True,
                cold_start=False,
                latency_saved_ms=120.0,
                input_tokens_saved=80,
            ),
            event(
                required_node_ids=("cli",),
                activated_node_ids=(),
                cache_eligible=True,
                cache_hit=False,
                cold_start=False,
            ),
        ]
    )

    assert summary.request_count == 3
    assert summary.required_node_count == 5
    assert summary.activated_required_node_count == 3
    assert summary.knowledge_node_coverage == 0.6

    assert summary.cache_eligible_count == 2
    assert summary.cache_hit_count == 1
    assert summary.cache_hit_rate == 0.5
    assert summary.cold_start_request_count == 1
    assert summary.warm_request_count == 2
    assert summary.warm_cache_hit_rate == 0.5

    assert summary.total_latency_saved_ms == 120.0
    assert summary.total_input_tokens_saved == 80


def test_event_rejects_cache_hit_when_lookup_was_not_eligible() -> None:
    with pytest.raises(ValidationError, match="cache hit"):
        KnowledgeUsageEvent(
            capsule_id="capsule-smoke-001",
            knowledge_tree_digest="a" * 64,
            request_id="request-001",
            task_family_id="cli-search",
            required_node_ids=(),
            activated_node_ids=(),
            cache_eligible=False,
            cache_hit=True,
            cold_start=False,
        )


def test_write_and_summarize_knowledge_usage_directory(tmp_path) -> None:
    config = TelemetryConfig(enabled=True, output_dir=tmp_path / "telemetry")

    first = write_knowledge_usage_event(
        config,
        capsule_path=tmp_path / "flight.zip",
        event=event(
            required_node_ids=("cli", "search"),
            activated_node_ids=("cli",),
            cache_eligible=True,
            cache_hit=True,
            cold_start=False,
            latency_saved_ms=25.0,
            input_tokens_saved=12,
        ),
    )
    second = write_knowledge_usage_event(
        config,
        capsule_path=tmp_path / "flight.zip",
        event=event(
            required_node_ids=("cli",),
            activated_node_ids=("cli",),
            cache_eligible=True,
            cache_hit=False,
            cold_start=False,
        ),
    )

    assert first is not None
    assert second is not None
    assert first != second
    assert first.parent == tmp_path / "telemetry" / "knowledge-usage"

    summary = summarize_knowledge_usage_directory(first.parent)

    assert summary.request_count == 2
    assert summary.knowledge_node_coverage == pytest.approx(2 / 3)
    assert summary.cache_hit_rate == 0.5
    assert summary.total_latency_saved_ms == 25.0
    assert summary.total_input_tokens_saved == 12


def test_write_knowledge_usage_can_be_disabled(tmp_path) -> None:
    config = TelemetryConfig(enabled=False, output_dir=Path("telemetry"))

    output = write_knowledge_usage_event(
        config,
        capsule_path=tmp_path / "flight.zip",
        event=event(
            required_node_ids=(),
            activated_node_ids=(),
            cache_eligible=False,
            cache_hit=False,
            cold_start=True,
        ),
    )

    assert output is None
    assert not (tmp_path / "telemetry").exists()


def result(
    *,
    relative_path: str = "docs/guide.md",
    chunk_index: int = 2,
    text: str = "capsule knowledge",
) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            relative_path=relative_path,
            source_type=SourceType.REPO,
            chunk_index=chunk_index,
            start_char=20,
            end_char=20 + len(text),
            text=text,
        ),
        score=0.9,
    )


def test_knowledge_node_id_is_stable_across_context_truncation() -> None:
    complete = result(text="capsule knowledge")
    truncated = result(text="capsule")

    complete_id = knowledge_node_id(complete.chunk)

    assert complete_id == knowledge_node_id(truncated.chunk)
    assert len(complete_id) == 64
    assert complete_id != knowledge_node_id(
        result(relative_path="docs/other.md").chunk
    )
    assert complete_id != knowledge_node_id(
        result(chunk_index=3).chunk
    )


def test_authored_node_reference_matches_runtime_chunk_identity() -> None:
    reference = KnowledgeNodeReference(
        relative_path="docs/guide.md",
        source_type=SourceType.REPO,
        chunk_index=2,
    )

    assert knowledge_node_reference_id(reference) == knowledge_node_id(
        result().chunk
    )


def test_build_retrieval_usage_event_records_observed_nodes() -> None:
    source = result()
    activated_id = knowledge_node_id(source.chunk)

    usage = build_retrieval_usage_event(
        capsule_id="capsule-smoke-001",
        knowledge_tree_digest="b" * 64,
        request_id="request-002",
        task_family_id="cli-search",
        sources=(source, source),
        required_node_ids=(activated_id, "missing-node"),
    )

    assert usage.activated_node_ids == (activated_id,)
    assert usage.required_node_ids == (activated_id, "missing-node")
    assert usage.cache_eligible is False
    assert usage.cache_hit is False
    assert usage.cold_start is True
