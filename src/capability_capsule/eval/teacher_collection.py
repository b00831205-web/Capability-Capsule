"""Preparation and append-only collection of Teacher trajectories."""

from pathlib import Path
from typing import Literal, Self

from pydantic import (
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
from capability_capsule.eval.student_target import StudentTarget, verify_student_target
from capability_capsule.eval.tasks import TaskSpec
from capability_capsule.eval.harness_profile import HarnessProfileReference, verify_harness_profile


class TeacherAssignment(BaseModel):
    """One authorized and reproducible Teacher trajectory assignment."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1", "0.2", "0.3"] = "0.2"
    assignment_id: str = Field(min_length = 1)
    trajectory_id: str = Field(min_length=1)
    teacher_model: str = Field(min_length=1)
    teacher_skill_version: str = Field(min_length=1)
    authorized_fixture_root: str = Field(min_length=1)
    destination_jsonl: Path
    task: TaskSpec
    student_target: StudentTarget | None = None
    harness_profile: HarnessProfileReference | None = None

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

        if self.schema_version == "0.1" and self.student_target is not None:
            raise ValueError("Schema 0.1 assignments cannot contain a Student target")

        if self.schema_version in {"0.1", "0.2"} and self.harness_profile is not None:
            raise ValueError(
                f"Schema {self.schema_version} assignment cannot contain a HarnessProfile"
            )

        if self.schema_version == "0.3" and self.harness_profile is None:
            raise ValueError(
                f"Schema 0.3 assignments require a HarnessProfile"
            )

        return self

def render_teacher_prompt(
    assignment: TeacherAssignment,
    *,
    artifact_root: Path | None = None,
) -> str:
    """Render a complete prompt for the Capsule Teacher skill."""

    if assignment.student_target is not None:
        verify_student_target(assignment.student_target, artifact_root=artifact_root)

    verified_harness_profile = None
    if assignment.harness_profile is not None:
        verified_harness_profile = verify_harness_profile(assignment.harness_profile, artifact_root = artifact_root)

    assignment_json = assignment.model_dump_json(indent=2)

    prompt = (
        "Use $capsule-teacher to execute this authorized Teacher assignment.\n"
        "Treat the JSON below as immutable provenance and constraints. "
        "Use only the authorized fixture and allowed tools. Capture only "
        "observable actions and validation evidence; do not record hidden "
        "reasoning.\n\n"
        f"{assignment_json}"
    )

    if verified_harness_profile is not None:
        prompt += (
            "\n\nVerified HarnessProfile\n"
            "Treat this as the exact Student-visible harness contract. "
            "Generate tool requests and observable tool results that conform "
            "to its tool schemas, shell semantics, and result envelopes.\n\n"
            f"{verified_harness_profile.model_dump_json(indent=2)}"
        )
    return prompt



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
        trajectory: TeacherTrajectory,
        *,
        artifact_root: Path | None = None,
) -> None:
    """Validate and append one completed Teacher trajectory."""

    _validate_assignment_match(assignment, trajectory)

    harness_profile = None
    if assignment.harness_profile is not None:
        harness_profile = verify_harness_profile(
            assignment.harness_profile,
            artifact_root = artifact_root,
        )

    validate_teacher_dataset(
        (trajectory,),
        tasks = (assignment.task,),
        harness_profile= harness_profile,
    )

    _reject_duplicate_trajectory_id(
        assignment.destination_jsonl,
        trajectory.trajectory_id,
    )

    append_jsonl(
        assignment.destination_jsonl,
        trajectory,
    )
