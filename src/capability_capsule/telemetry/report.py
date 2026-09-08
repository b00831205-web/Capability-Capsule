"""Summarize local capsule runtime telemetry."""

from collections import Counter
from math import ceil
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.telemetry.writer import RunTelemetryEvent


class TelemetryReport(BaseModel):
    """Aggregate metrics calculated from local runtime events"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: Path
    event_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    mean_duration_ms: float = Field(ge=0)
    p95_duration_ms: float = Field(ge=0)
    total_question_chars: int = Field(ge=0)
    total_source_count: int = Field(ge=0)
    error_types: dict[str, int]
    capsule_build_ids: tuple[UUID, ...]
    generation_models: tuple[str, ...]


def _nearest_rank_p95(values: list[float]) -> float:
    """Calculate the nearest-rank 95th percentile."""

    if not values:
        return 0.0

    ordered = sorted(values)
    rank = max(1, ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def summarize_telemetry(output_dir: Path) -> TelemetryReport:
    """Load local telemetry events and calculate aggregate metrics"""

    source = output_dir.resolve(strict=True)

    if not source.is_dir():
        raise NotADirectoryError(source)

    events = [
        RunTelemetryEvent.model_validate_json(path.read_bytes())
        for path in sorted(source.glob("*.json"))
        if path.is_file()
    ]

    event_count = len(events)
    success_count = sum(event.status == "success" for event in events)
    error_count = sum(event.status == "error" for event in events)
    durations = [event.duration_ms for event in events]

    error_types = Counter(
        event.error_type or "UnknownError" for event in events if event.status == "error"
    )

    build_ids = tuple(
        sorted(
            {event.capsule_build_id for event in events},
            key=str,
        )
    )

    models = tuple(sorted({event.generation_model for event in events}))

    return TelemetryReport(
        output_dir=source,
        event_count=event_count,
        success_count=success_count,
        error_count=error_count,
        success_rate=success_count / event_count if event_count else 0.0,
        mean_duration_ms=(sum(durations) / event_count if event_count else 0.0),
        p95_duration_ms=_nearest_rank_p95(durations),
        total_question_chars=sum(event.question_chars for event in events),
        total_source_count=sum(event.source_count for event in events),
        error_types=dict(sorted(error_types.items())),
        capsule_build_ids=build_ids,
        generation_models=models,
    )
