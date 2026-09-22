"""Reproducible and resumable batch Teacher collection plans."""

from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capability_capsule.eval.dataset_validation import (
    validate_teacher_dataset,
)
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    TeacherTrajectory,
)
from capability_capsule.eval.student_target import StudentTarget
from capability_capsule.eval.tasks import TaskSpec
from capability_capsule.eval.teacher_collection import TeacherAssignment
from capability_capsule.eval.harness_profile import HarnessProfileReference, HarnessProfile, verify_harness_profile


class TeacherCollectionPlan(BaseModel):
    """An ordered batch of authorized Teacher assignments."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1", "0.2", "0.3"] = "0.2"
    plan_id: str = Field(min_length=1)
    assignments: tuple[TeacherAssignment, ...] = Field(min_length=1)
    student_target: StudentTarget | None = None
    harness_profile: HarnessProfileReference | None = None

    @field_validator("plan_id")
    @classmethod
    def reject_blank_plan_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plan_id must not be blank")

        return value

    @model_validator(mode="after")
    def validate_assignment(self)-> Self:
        assignment_ids: set[str] = set()
        trajectory_ids: set[str] = set()
        task_ids: set[str] = set()
        split_by_group: dict[str, DatasetSplit] = {}

        if self.schema_version == "0.1" and self.student_target is not None:
            raise ValueError("Schema 0.1 plans cannot contain a Student target")

        if self.schema_version in {"0.1", "0.2"} and self.harness_profile is not None:
            raise ValueError(
                f"Schema {self.schema_version} plans cannot contain a HarnessProfile"
            )
        if self.schema_version =="0.3" and self.harness_profile is None:
            raise ValueError(
                "Schema 0.3 plans require a HarnessProfile"
            )

        for assignment in self.assignments:
            if assignment.assignment_id in assignment_ids:
                raise ValueError(
                    f"Duplicate assignment ID "
                    f"{assignment.assignment_id!r}"
                )

            assignment_ids.add(assignment.assignment_id)

            if assignment.trajectory_id in trajectory_ids:
                raise ValueError(
                    f"Duplicate trajectory ID "
                    f"{assignment.trajectory_id!r}"
                )

            trajectory_ids.add(assignment.trajectory_id)

            task = assignment.task

            if task.task_id in task_ids:
                raise ValueError(
                    f"Duplicate task ID {task.task_id!r}"
                )

            task_ids.add(task.task_id)

            if task.split is DatasetSplit.TEST:
                raise ValueError(
                    "Teacher collection plans must not contain "
                    "the locked test split"
                )

            group = task.split_group
            previous_split = split_by_group.get(group)

            if (
                previous_split is not None
                and previous_split is not task.split
            ):
                raise ValueError(
                    f"Split leakage for group {group!r}: "
                    f"{previous_split.value}, {task.split.value}"
                )
            split_by_group[group] = task.split

            if assignment.student_target != self.student_target:
                raise ValueError(
                    "Every assignment must reference the plan's Student target"
                )

            if assignment.harness_profile != self.harness_profile:
                raise ValueError(
                    "Every assignment must reference the plan's HarnessProfile"
                )

        return self

def build_teacher_collection_plan(
        *,
        plan_id: str,
        tasks: Sequence[TaskSpec],
        teacher_model: str,
        teacher_skill_version: str,
        fixture_roots: Mapping[str, str],
        destination_jsonl: Path,
        trajectory_ids: Mapping[str, str],
        student_target: StudentTarget | None = None,
        harness_profile: HarnessProfileReference | None = None,
) -> TeacherCollectionPlan:
    """Build an ordered collection plan from explicit provenance"""

    schema_version: Literal["0.2", "0.3"] = "0.3" if harness_profile is not None else "0.2"
    assignments: list[TeacherAssignment] = []

    for task in tasks:
        try:
            fixture_root = fixture_roots[task.fixture_id]
        except KeyError as error:
            raise ValueError(
                f"Missing fixture root for fixture "
                f"{task.fixture_id!r}"
            ) from error

        try:
            trajectory_id = trajectory_ids[task.task_id]
        except KeyError as error:
            raise ValueError(
                f"Missing trajectory ID for task {task.task_id!r}"
            ) from error

        assignments.append(
            TeacherAssignment(
                schema_version=schema_version,
                assignment_id = f"{plan_id}:{task.task_id}",
                trajectory_id = trajectory_id,
                teacher_model = teacher_model,
                teacher_skill_version = teacher_skill_version,
                authorized_fixture_root = fixture_root,
                destination_jsonl = destination_jsonl,
                task = task,
                student_target=student_target,
                harness_profile= harness_profile,
            )
        )

    return TeacherCollectionPlan(
        schema_version=schema_version,
        plan_id = plan_id,
        assignments= tuple(assignments),
        student_target=student_target,
        harness_profile=harness_profile,
    )

def _load_trajectory_index(
        destination: Path,
) -> dict[str, TeacherTrajectory]:
    if not destination.exists():
        return {}

    trajectories = load_jsonl(
        destination,
        TeacherTrajectory,
    )
    index: dict[str, TeacherTrajectory] = {}

    for trajectory in trajectories:
        if trajectory.trajectory_id in index:
            raise ValueError(
                f"Duplicate trajectory ID "
                f"{trajectory.trajectory_id!r} in {destination}"
            )

        index[trajectory.trajectory_id] = trajectory

    return index


def _resolve_destination_path(
    path: Path,
    *,
    artifact_root: Path | None,
) -> Path:
    if path.is_absolute() or artifact_root is None:
        return path

    raw_path = str(path)
    windows_path = PureWindowsPath(raw_path)
    root = Path(artifact_root)

    if windows_path.is_absolute():
        relative_parts = list(windows_path.parts[1:])
        root_name = root.name.casefold()
        for index, part in enumerate(relative_parts):
            if part.casefold() == root_name:
                return root.joinpath(*relative_parts[index + 1 :])
        return path

    if "\\" in raw_path:
        return root.joinpath(*windows_path.parts)

    return root / path

def _validate_completed_assignment(
        assignment: TeacherAssignment,
        trajectory: TeacherTrajectory,
        *,
        harness_profile: HarnessProfile | None = None
) -> None:
    if trajectory.teacher_model != assignment.teacher_model:
        raise ValueError(
            f"Collected trajectory {trajectory.trajectory_id!r} "
            "has the wrong Teacher model"
        )

    if (
        trajectory.teacher_skill_version != assignment.teacher_skill_version
    ): 
        raise ValueError(
            f"Collected trajectory {trajectory.trajectory_id!r} "
            "has wrong skill version"
        )

    validate_teacher_dataset(
        (trajectory,),
        tasks = (assignment.task,),
        harness_profile= harness_profile,
    )

def pending_teacher_assignments(
        plan: TeacherCollectionPlan,
        *,
        artifact_root: Path | None = None
) -> tuple[TeacherAssignment, ...]:
    """Return assignments not yet present in their raw JSONL destinations."""

    harness_profile = None
    if plan.harness_profile is not None:
        harness_profile = verify_harness_profile(plan.harness_profile, artifact_root=artifact_root)
    index_by_destination: dict[Path, dict[str, TeacherTrajectory]] = {}
    pending: list[TeacherAssignment] = []

    for assignment in plan.assignments:
        destination = _resolve_destination_path(
            assignment.destination_jsonl,
            artifact_root=artifact_root,
        )

        if destination not in index_by_destination:
            index_by_destination[destination] = (
                _load_trajectory_index(destination)
            )

        existing = index_by_destination[destination].get(
            assignment.trajectory_id
        )

        if existing is None:
            pending.append(assignment)
            continue

        _validate_completed_assignment(
            assignment,
            existing,
            harness_profile = harness_profile
        )

    return tuple(pending)
