"""Write privacy-preserving local runtime telemetry"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.config import TelemetryConfig


class RunTelemetryEvent(BaseModel):
    """Local metadata describing one capsule run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capsule_build_id: UUID
    generation_model: str = Field(min_length=1)
    duration_ms: float = Field(ge=0)
    question_chars: int = Field(ge=0)
    source_count: int = Field(ge=0)
    status: Literal["success", "error"]
    error_type: str | None = None


def write_run_telemetry(
    config: TelemetryConfig,
    *,
    capsule_path: Path,
    capsule_build_id: UUID,
    generation_model: str,
    duration_ms: float,
    question_chars: int,
    source_count: int,
    status: Literal["success", "error"],
    error_type: str | None = None,
) -> Path | None:
    """Write one local telemetry event, or do nothing when disabled"""

    if not config.enabled:
        return None

    output_dir = config.output_dir
    if not output_dir.is_absolute():
        output_dir = capsule_path.resolve().parent / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    event = RunTelemetryEvent(
        capsule_build_id=capsule_build_id,
        generation_model=generation_model,
        duration_ms=duration_ms,
        question_chars=question_chars,
        source_count=source_count,
        status=status,
        error_type=error_type,
    )

    output_path = output_dir / f"{event.event_id}.json"

    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(event.model_dump_json(indent=2))
        stream.write("\n")

    return output_path
