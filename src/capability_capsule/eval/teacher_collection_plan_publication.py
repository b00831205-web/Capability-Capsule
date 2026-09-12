"""Immutable publication of Teacher collection plans."""


import shutil
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from capability_capsule.eval.teacher_collection_plan import(
    TeacherCollectionPlan,
)


_PLAN_FILENAME = "collection-plan.json"
_MANIFEST_FILENAME = "manifest.json"
_EXPECTED_FILENAMES = {
    _PLAN_FILENAME,
    _MANIFEST_FILENAME,
}


def _validate_plan_id(plan_id: str) -> str:
    if (
        not plan_id.strip()
        or plan_id != plan_id.strip()
        or plan_id in {".", ".."}
        or "/" in plan_id
        or "\\" in plan_id
        or "\0" in plan_id
    ):
        raise ValueError(
            "plan_id must be a safe, non-blank directory name"
        )

    return plan_id

class PublishedTeacherCollectionPlan(BaseModel):
    """Portable integrity record for one immutable collection plan."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    plan_id: str = Field(min_length=1)
    plan_filename: Literal["collection-plan.json"] = "collection-plan.json"
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern = r"^[0-9a-f]{64}$")
    plan : TeacherCollectionPlan

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return _validate_plan_id(value)


def _plan_bytes(plan: TeacherCollectionPlan) -> bytes:
    return plan.model_dump_json(indent =2).encode("utf-8") +b"\n"

def publish_teacher_collection_plan(
        plan: TeacherCollectionPlan,
        *,
        output_root: Path
) -> PublishedTeacherCollectionPlan:
    """Publish one plan without overwriting an existing plan version."""

    validated_plan_id = _validate_plan_id(plan.plan_id)
    output_root = Path(output_root)
    plan_dir = output_root / validated_plan_id

    if plan_dir.exists():
        raise FileExistsError(
            f"Collection plan {validated_plan_id!r} already exists"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        mkdtemp(
            prefix = f".{validated_plan_id}.",
            dir = output_root,
        )
    )

    try:
        plan_bytes = _plan_bytes(plan)
        plan_path = staging_dir / _PLAN_FILENAME
        plan_path.write_bytes(plan_bytes)

        published = PublishedTeacherCollectionPlan(
            plan_id = validated_plan_id,
            byte_count=len(plan_bytes),
            sha256 = sha256(plan_bytes).hexdigest(),
            plan = plan
        )

        manifest_bytes = (
            published.model_dump_json(indent = 2).encode("utf-8") + b"\n"
        )
        (staging_dir / _MANIFEST_FILENAME).write_bytes(
            manifest_bytes
        )

        staging_dir.rename(plan_dir)
        return published

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

def _validate_file_set(plan_dir: Path) -> None:
    actual = {entry.name for entry in plan_dir.iterdir()}
    missing = _EXPECTED_FILENAMES - actual
    unexpected = actual - _EXPECTED_FILENAMES

    if missing:
        raise ValueError(
            "Published collection plan has missing files: "
            f"{', '.join(sorted(missing))}"
        )

    if unexpected:
        raise ValueError(
            "Published collection plan has unexpected files: "
            f"{', '.join(sorted(unexpected))}"
        )

def load_teacher_collection_plan(
        plan_dir: Path
) -> PublishedTeacherCollectionPlan:
    """Load and verify an immutable Teacher collection plan."""

    plan_dir = Path(plan_dir)
    if not plan_dir.is_dir():
        raise ValueError(
            f"Collection plan directory does not exist: {plan_dir}"
        )

    _validate_file_set(plan_dir)

    manifest_path = plan_dir / _MANIFEST_FILENAME

    try:
        published = PublishedTeacherCollectionPlan.model_validate_json(
            manifest_path.read_text(encoding = "utf-8")
        )

    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid collection plan manifest: {manifest_path}"
        )

    if published.plan_id != plan_dir.name:
        raise ValueError(
            f"Manifest plan_id {published.plan_id!r} does not match "
            f"directory name {plan_dir.name!r}"
        )

    plan_path = plan_dir / published.plan_filename
    payload = plan_path.read_bytes()
    actual_digest = sha256(payload).hexdigest()

    if actual_digest != published.sha256:
        raise ValueError(
            f"{published.plan_filename} SHA-256 mismatch: "
            f"expected {published.sha256}, got {actual_digest}"
        )

    if len(payload) != published.byte_count:
        raise ValueError(
            f"{published.plan_filename} byte count mismatch: "
            f"expected {published.byte_count}, got {len(payload)}"
        ) 

    try:
        loaded_plan = TeacherCollectionPlan.model_validate_json(payload)

    except ValueError as error:
        raise ValueError(
            f"Invalid collection plan: {plan_path}"
        ) from error

    if loaded_plan != published.plan:
        raise ValueError(
            "Collection plan does not match the manifest copy"
        )

    return published