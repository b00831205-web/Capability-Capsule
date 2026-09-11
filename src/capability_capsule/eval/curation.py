"""Deterministic curation helpers for Teacher trajectory datasets."""

import json
import unicodedata
from collections.abc import Mapping, Sequence
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory


class DuplicateTrajectory(BaseModel):
    """One duplicate and the earlier trajectory retained in the dataset."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    kept_trajectory_id: str = Field(min_length =1)
    duplicate_trajectory_id: str = Field(min_length=1)

class DeduplicationResult(BaseModel):
    """Stable deduplication output and its audit report."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    unique_trajectories: tuple[TeacherTrajectory, ...]
    duplicates: tuple[DuplicateTrajectory, ...]

def _normalize_task(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())

def _normalize_content(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r","\n")

    lines = [line.rstrip() for line in normalized.split("\n")]

    while lines and not lines[0]:
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)

def trajectory_fingerprint(trajectory: TeacherTrajectory) -> str:
    """Return a stable content fingerprint independent of IDs and split."""

    payload = {
        "task": _normalize_task(trajectory.task),
        "messages": [
            {
                "role": message.role.value,
                "content": _normalize_content(message.content),
                "tool_calls": [
                    {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in message.tool_calls
                ],
                "tool_name": message.tool_name,
            }
            for message in trajectory.messages
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()

def deduplicate_trajectories(
        trajectories: Sequence[TeacherTrajectory],
) -> DeduplicationResult:
    """Keep the first occurrence of each fingerprint and report duplicates"""

    first_by_fingerprint: dict[str, TeacherTrajectory] = {}
    unique: list[TeacherTrajectory] = []
    duplicates: list[DuplicateTrajectory] = []

    for trajectory in trajectories:
        fingerprint = trajectory_fingerprint(trajectory)
        first = first_by_fingerprint.get(fingerprint)

        if first is None:
            first_by_fingerprint[fingerprint] = trajectory
            unique.append(trajectory)
            continue

        duplicates.append(
            DuplicateTrajectory(
                fingerprint = fingerprint,
                kept_trajectory_id = first.trajectory_id,
                duplicate_trajectory_id= trajectory.trajectory_id,
            )
        )

    return DeduplicationResult(
        unique_trajectories = tuple(unique),
        duplicates = tuple(duplicates),
    )

def partition_trajectories(
        trajectories: Sequence[TeacherTrajectory],
) -> dict[DatasetSplit, tuple[TeacherTrajectory, ...]]:
    """Partition trajectories by their preassigned split."""

    partitions: dict[DatasetSplit, list[TeacherTrajectory]] = {
        split: [] for split in DatasetSplit
    }

    for trajectory in trajectories:
        partitions[trajectory.split].append(trajectory)
    return {
        split: tuple(records)
        for split, records in partitions.items()
    }

def validate_split_isolation(
        trajectories: Sequence[TeacherTrajectory],
        *,
        group_by_task: Mapping[str, str]
) -> None:
    """Reject fixture or defect-family groups that cross dataset splits"""

    splits_by_group: dict[str, set[DatasetSplit]] = {}

    for trajectory in trajectories:
        try:
            group = group_by_task[trajectory.task_id]

        except KeyError as error:
            raise ValueError(
                f"Missing split group for task {trajectory.task_id!r}"
            ) from error

        if not group.strip():
            raise ValueError(
                f"Split group for task {trajectory.task_id!r} must not be blank"
            )

        group_splits = splits_by_group.setdefault(group, set())
        group_splits.add(trajectory.split)

        if len(group_splits) >1:
            split_names = ", ".join(
                sorted(split.value for split in group_splits)
            )
            raise ValueError(
                f"Split leakage for group {group!r}: {split_names}"
            )