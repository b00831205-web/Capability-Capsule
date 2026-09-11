"""Validated and auditable Teacher dataset curation pipeline."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from capability_capsule.eval.curation import (
    DuplicateTrajectory,
    deduplicate_trajectories,
    partition_trajectories,
)
from capability_capsule.eval.dataset_validation import(
    validate_teacher_dataset,
)
from capability_capsule.eval.records import (
    DatasetSplit,
    TeacherTrajectory,
)
from capability_capsule.eval.tasks import TaskSpec


class CuratedTeacherDataset(BaseModel):
    """Validated training partition and their duplicate audit report."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    train: tuple[TeacherTrajectory, ...]
    validation: tuple[TeacherTrajectory, ...]
    duplicates: tuple[DuplicateTrajectory, ...]

def curate_teacher_dataset(
        trajectories: Sequence[TeacherTrajectory],
        *,
        tasks: Sequence[TaskSpec],
) -> CuratedTeacherDataset:
    """Validate, deduplicate, and partition raw Teacher trajectories."""

    validate_teacher_dataset(
        trajectories,
        tasks = tasks
    )

    deduplication = deduplicate_trajectories(trajectories)
    partitions = partition_trajectories(
        deduplication.unique_trajectories
    )

    return CuratedTeacherDataset(
        train = partitions[DatasetSplit.TRAIN],
        validation = partitions[DatasetSplit.VALIDATION],
        duplicates=deduplication.duplicates,
    )