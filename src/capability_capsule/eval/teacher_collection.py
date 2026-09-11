"""Preparation and append-only collection of Teacher trajectories."""

from pathlib import Path
from typing import Literal, Self

from pydantic import(
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from capability_capsule.eval.dataset_validation import (
    validate_teacher_dataset,
)
from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    TeacherTrajectory,
)
from capability_capsule.eval.tasks import TaskSpec


class TeacherAssignment(BaseModel):
    """One authorized and reproducible Teacher trajectory assignment."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    assignment_id: str = Field(min_length = 1)
    trajectory_id: str = Field(min_length=1)
    teacher_model: str = Field(min_length=1)
    teacher_skill_version: str = Field(min_length=1)
    authorized_fixture_root: str = Field(min_length=1)
    destination_jsonl: Path
    task: TaskSpec

    @field_validator(
        "assignment_id",
        "trajectory_id",
        "teacher_model",
        "teacher_skill_version",
        "authorized_fixture_root",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Assignment text fields must not be blank")

        return value

    @field_validator("destination_jsonl")
    @classmethod
    def require_jsonl_destination(cls, value: Path) -> Path:
        if value.suffix.lower() != ".jsonl":
            raise ValueError(
                "destination_jsonl must use the .jsonl extension"
            )
        return value

    @model_validator(mode="after")
    def reject_locked_test_split(self) -> Self:
        if self.task.split is DatasetSplit.TEST:
            raise ValueError(
                "Teacher assignments must not use the locked test split"
            )

        return self

def render_teacher_prompt(assignment: TeacherAssignment) -> str:
    """Render a complete prompt for the Capsule Teacher skill."""

    assignment_json = assignment.model_dump_json(indent=2)

    return (
        "Use $capsule-teacher to execute this authorized Teacher assignment.\n"
        "Treat the JSON below as immutable provenance and constraints. "
        "Use only the authorized fixture and allowed tools. Capture only "
        "observable actions and validation evidence; do not record hidden "
        "reasoning.\n\n"
        f"{assignment_json}"
    )

def _validate_assignment_match(
        assignment: TeacherAssignment,
        trajectory: TeacherTrajectory,
) -> None:
    if trajectory.trajectory_id != assignment.trajectory_id:
        raise ValueError(
            "Trajectory ID does not match the Teacher assignment"
        )

    if trajectory.teacher_model != assignment.teacher_model:
        raise ValueError(
            "Teacher model does not match the Teacher assignment"
        )

    if (
        trajectory.teacher_skill_version != assignment.teacher_skill_version
    ):
        raise ValueError(
            "Teacher skill version does not match the Teacher assignment"
        )


def _reject_duplicate_trajectory_id(
        destination: Path,
        trajectory_id: str
) -> None:
    if not destination.exists():
        return 

    existing = load_jsonl(destination, TeacherTrajectory)

    if any(
        trajectory.trajectory_id == trajectory_id for trajectory in existing
    ):
        raise ValueError(
            f"Trajectory ID {trajectory_id!r} already exists in "
            f"{destination}"
        )


def append_teacher_trajectory(
        assignment: TeacherAssignment,
        trajectory: TeacherTrajectory
) -> None:
    """Validate and append one completed Teacher trajectory."""

    _validate_assignment_match(assignment, trajectory)

    validate_teacher_dataset(
        (trajectory,),
        tasks = (assignment.task,),
    )

    _reject_duplicate_trajectory_id(
        assignment.destination_jsonl,
        trajectory.trajectory_id,
    )

    append_jsonl(
        assignment.destination_jsonl,
        trajectory,
    )