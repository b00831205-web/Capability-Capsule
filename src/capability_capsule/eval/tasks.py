"""Versioned task specification and logical experiment cycles"""

from datetime import timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from capability_capsule.eval.records import DatasetSplit


class TaskCategory(StrEnum):
    """Pre-registered coding-task category"""

    CODE_SEARCH = "code_search"
    SINGLE_FILE_CHANGE = "single_file_change"
    MULTI_FILE_CHANGE = "multi_file_change"
    FAILURE_DIAGNOSIS = "failure_diagnosis"
    TOOL_ERROR_RECOVERY = "tool_error_recovery"


class TaskDifficulty(StrEnum):
    """Pre-registered task difficulty band"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ChangeOperation(StrEnum):
    """Expected filesystem operation for one task"""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"

class ExpectedChange(BaseModel):
    """One allowed and expected fixture-relative file change"""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    path: str = Field(min_length=1)
    operation: ChangeOperation

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Expected change path must not be blank")

        normalized = value.replace("\\","/")
        path = PurePosixPath(normalized)

        if (
            path.is_absolute()
            or ".." in path.parts
            or path == PurePosixPath(".")
            or (path.parts and path.parts[0].endswith(":"))
        ):
            raise ValueError(
                "Expected change path must stay inside the fixture"
            )

        return value

class TaskSpec(BaseModel):
    """One immutable task definition used by training or evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    task_id: str = Field(min_length=1)
    fixture_id: str = Field(min_length=1)
    fixture_family_id: str = Field(min_length=1)
    fixture_revision: str = Field(min_length=1)
    defect_family: str = Field(min_length=1)
    split: DatasetSplit
    category: TaskCategory
    difficulty: TaskDifficulty
    task: str = Field(min_length=1)
    knowledge_distance: float = Field(ge=0.0)
    logical_arrival: timedelta = Field(ge=timedelta(0))
    time_limit: timedelta = Field(gt=timedelta(0))
    allowed_tools: tuple[str, ...] = Field(min_length=1)
    expected_changes: tuple[ExpectedChange, ...] = ()
    validation_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator(
        "task_id",
        "fixture_id",
        "fixture_family_id",
        "fixture_revision",
        "defect_family",
        "task",
    )
    @classmethod
    def reject_blank_required_text(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("Required task text must not be blank")

        return value

    @field_validator("allowed_tools", "validation_ids")
    @classmethod
    def reject_blank_collection_values(
        cls,
        value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("Task collection values must not be blank")

        return value

    @property
    def split_group(self) -> str:
        """Return the fixture-family and defect-family leakage boundary"""

        return f"{self.fixture_family_id}:{self.defect_family}"


class TaskCycle(BaseModel):
    """One orderd and indivisible logical task-event cycle"""

    model_config = ConfigDict(extra="forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    cycle_id: str = Field(min_length=1)
    split: DatasetSplit
    tasks: tuple[TaskSpec, ...] = Field(min_length=1)

    @field_validator("cycle_id")
    @classmethod
    def reject_blank_cycle_id(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("cycle_id must not be blank")

        return value

    @model_validator(mode="after")
    def validate_cycle(self) -> Self:
        task_ids = [task.task_id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique within a cycle")

        if any(task.split is not self.split for task in self.tasks):
            raise ValueError(
                "Every task in a cycle must use the cycle split"
            )

        arrivals = [task.logical_arrival for task in self.tasks]
        if arrivals != sorted(arrivals):
            raise ValueError(
                "Tasks must be order by nondecreasing logical arrival"
            )

        return self
