"""Summarize knowledge-tree coverage and cache reuse events"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Self
from uuid import UUID, uuid4
from hashlib import sha256
import json

from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult

from pathlib import Path

from capability_capsule.config import TelemetryConfig
from capability_capsule.manifest import SourceType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeNodeReference(BaseModel):
    """Authorable reference to one knowledge node in an index."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    relative_path: str = Field(min_length=1)
    source_type: SourceType
    chunk_index: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def reject_blank_path(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("Knowledge node path must not be blank")

        return value

class KnowledgeUsageEvent(BaseModel):
    """One observable knowledge-tree and cache-use decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    event_id: UUID = Field(default_factory=uuid4)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capsule_id: str = Field(min_length=1)
    knowledge_tree_digest: str = Field(pattern = r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1)
    task_family_id: str = Field(min_length=1)
    required_node_ids: tuple[str, ...]
    activated_node_ids: tuple[str, ...]
    cache_eligible: bool
    cache_hit: bool
    cold_start: bool
    latency_saved_ms: float = Field(default=0.0, ge=0.0)
    input_tokens_saved: int = Field(default=0, ge=0)

    @field_validator(
        "capsule_id",
        "request_id",
        "task_family_id",
    )
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifiers must not be blank")

        return value

    @field_validator(
        "required_node_ids",
        "activated_node_ids",
    )
    @classmethod
    def validate_node_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not node_id.strip() for node_id in value):
            raise ValueError("Knowledge node identifiers must not be blank")

        if len(set(value)) != len(value):
            raise ValueError("Knowledge mnode identifiers must be unique per event")

        return value

    @model_validator(mode = "after")
    def validate_cache_state(self) -> Self:
        if self.cache_hit and not self.cache_eligible:
            raise ValueError("A cache hit requires an eligible cache lookup")

        return self

class KnowledgeUsageSummary(BaseModel):
    """Aggregate converage and cache-reuse metrics across runtime events"""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    request_count: int = Field(ge=0)
    required_node_count: int = Field(ge=0)
    activated_required_node_count: int = Field(ge=0)
    knowledge_node_coverage: float = Field(ge= 0.0, le=1.0)
    cache_eligible_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    cold_start_request_count: int = Field(ge=0.0)
    warm_request_count: int = Field(ge=0)
    warm_cache_eligible_count: int = Field(ge=0)
    warm_cache_hit_count: int = Field(ge=0)
    warm_cache_hit_rate: float = Field(ge=0.0, le=1.0)
    total_latency_saved_ms: float = Field(ge = 0.0)
    total_input_tokens_saved: int = Field(ge=0)

def knowledge_node_reference_id(
        reference: KnowledgeNodeReference,
) -> str:
    """Return a stable opaque identifier for one indexed knowledge node."""

    identity = {
        "relative_path": reference.relative_path,
        "source_type": reference.source_type.value,
        "chunk_index": reference.chunk_index,
    }

    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256(canonical).hexdigest()

def knowledge_node_id(chunk: TextChunk) -> str:
    """Return a stable opaque identifier for one indexed knowledge node."""

    reference = KnowledgeNodeReference(
        relative_path = chunk.relative_path,
        source_type = chunk.source_type,
        chunk_index = chunk.chunk_index
    )
    return knowledge_node_reference_id(reference)

def build_retrieval_usage_event(
        *,
        capsule_id: str,
        knowledge_tree_digest: str,
        request_id: str,
        sources: Sequence[SearchResult],
        task_family_id: str = "unlabeled-runtime",
        required_node_ids: tuple[str, ...] =(),
        cache_eligible: bool = False,
        cache_hit: bool = False,
        cold_start: bool = True,
        latency_saved_ms: float = 0.0,
        input_tokens_saved: int = 0,
) -> KnowledgeUsageEvent:
    """Build an event from nodes actually supplied to generation."""

    activated_node_ids = tuple(
        dict.fromkeys(
            knowledge_node_id(
                source.chunk
            )
            for source in sources
        )
    )

    return KnowledgeUsageEvent(
        capsule_id = capsule_id,
        knowledge_tree_digest=knowledge_tree_digest,
        request_id = request_id,
        task_family_id= task_family_id,
        required_node_ids= required_node_ids,
        activated_node_ids= activated_node_ids,
        cache_eligible=cache_eligible,
        cache_hit= cache_hit,
        cold_start = cold_start,
        latency_saved_ms= latency_saved_ms,
        input_tokens_saved= input_tokens_saved
    )

def summarize_knowledge_usage(
        events: Sequence[KnowledgeUsageEvent]
) -> KnowledgeUsageSummary:
    """Calculate coverage seperately from cache-reuse metrics"""

    required_node_count = sum(
        len(event.required_node_ids)
        for event in events
    )

    activated_required_node_count = sum(
        len(set(event.required_node_ids) & set(event.activated_node_ids)) for event in events
    )

    cache_eligible_events = [
        event
        for event in events if event.cache_eligible
    ]

    warm_events = [event for event in events if not event.cold_start]
    
    warm_cache_eligible_events = [event for event in warm_events if event.cache_eligible]

    cache_hit_count = sum(event.cache_hit for event in cache_eligible_events)

    warm_cache_hit_count = sum(event.cache_hit for event in warm_cache_eligible_events)

    return KnowledgeUsageSummary(
        request_count = len(events),
        required_node_count= required_node_count,
        activated_required_node_count= activated_required_node_count,
        knowledge_node_coverage=(
            activated_required_node_count / required_node_count
            if required_node_count else 0.0
        ),
        cache_eligible_count = len(cache_eligible_events),
        cache_hit_count = cache_hit_count,
        cache_hit_rate = (
            cache_hit_count / len(cache_eligible_events)
            if cache_eligible_events else 0.0
        ),
        cold_start_request_count= sum(
            event.cold_start for event in events
        ),
        warm_request_count = len(warm_events),
        warm_cache_eligible_count= len(warm_cache_eligible_events),
        warm_cache_hit_count= warm_cache_hit_count,
        warm_cache_hit_rate = (
            warm_cache_hit_count / len(warm_cache_eligible_events)
            if warm_cache_eligible_events
            else 0.0
        ),
        total_latency_saved_ms= sum(
            event.latency_saved_ms 
            for event in events if event.cache_hit
        ),
        total_input_tokens_saved = sum(
            event.input_tokens_saved for event in events if event.cache_hit
        )
    )

def write_knowledge_usage_event(
        config: TelemetryConfig,
        *,
        capsule_path: Path,
        event: KnowledgeUsageEvent,
) -> Path | None:
    """Write one local knowledge-usage event when telemetry is enabled."""

    if not config.enabled:
        return None

    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = capsule_path.resolve().parent / output_dir

    output_dir = output_dir / "knowledge-usage"
    output_dir.mkdir(parents = True, exist_ok = True)

    output_path = output_dir / f"{event.event_id}.json"

    with output_path.open("x", encoding = "utf-8") as stream:
        stream.write(event.model_dump_json(indent=2))
        stream.write("\n")

    return output_path

def summarize_knowledge_usage_directory(
        output_dir: Path,
) -> KnowledgeUsageSummary:
    """Load persisted knowledge-usage events and summarize them."""

    source = output_dir.resolve(strict = True)

    if not source.is_dir():
        raise NotADirectoryError(source)

    events = [
        KnowledgeUsageEvent.model_validate_json(path.read_bytes())
        for path in sorted(source.glob("*.json"))
        if path.is_file()
    ]

    return summarize_knowledge_usage(events)