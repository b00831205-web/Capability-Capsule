import pytest
from pydantic import ValidationError

from capability_capsule.telemetry.knowledge_usage import (
    KnowledgeUsageEvent,
    summarize_knowledge_usage,
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
