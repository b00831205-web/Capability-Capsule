"""Stable content provenance for authorized Teacher fixtures."""


import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from capability_capsule.eval.tasks import TaskSpec


_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
_IGNORED_FILE_SUFFIXES = (".pyc",)

class FixtureSnapshot(BaseModel):
    """Portable content identity for one fixture directory"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    fixture_id: str = Field(min_length=1)
    revision: str = Field(pattern = r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=1)
    byte_count: int = Field(ge=0)

    @field_validator("fixture_id")
    @classmethod
    def reject_blank_fixture_id(cls, value:str) -> str:
        if not value.strip():
            raise ValueError("fixture_id must not be blank")

        return value

def _normalized_relative_path(
        root: Path,
        path: Path,
) -> str:
    relative = path.relative_to(root).as_posix()
    return unicodedata.normalize("NFC", relative)

def _collect_fixture_files(
        root: Path
) -> tuple[tuple[str, Path], ...]:
    if not root.exists():
        raise ValueError(
            f"Fixture directory does not exist: {root}"
        )

    if not root.is_dir():
        raise ValueError(
            f"Fixture path is not a directory: {root}"
        )

    files: list[tuple[str, Path]] = []
    normalized_paths: set[str] = set()

    for path in root.rglob("*"):
        relative_path = path.relative_to(root)

        if any(
            part in _IGNORED_DIRECTORY_NAMES
            for part in relative_path.parts
        ):
            continue

        if path.name.endswith(_IGNORED_FILE_SUFFIXES):
            continue

        relative = _normalized_relative_path(root, path)

        if path.is_symlink():
            raise ValueError(
                f"Fixture contains symbolic link {relative!r}"
            )

        if path.is_dir():
            continue

        if not path.is_file():
            raise ValueError(
                f"Fixture contains unsupported entry {relative!r}"
            )

        if relative in normalized_paths:
            raise ValueError(
                f"Fixture contains duplicate normalized path "
                f"{relative!r}"
            )

        normalized_paths.add(relative)
        files.append((relative, path))

    if not files:
        raise ValueError(
            "Fixture must contain at least one file"
        )

    files.sort(key = lambda item: item[0])
    return tuple(files)

def inspect_fixture(
        fixture_id: str,
        root: Path
) -> FixtureSnapshot:
    """Calculate a stable identity from fixture paths and raw bytes."""

    root = Path(root)
    files = _collect_fixture_files(root)
    digest = sha256()
    total_byte_count = 0

    for relative, path in files:
        relative_bytes = relative.encode("utf-8")
        expected_size = path.stat().st_size

        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(expected_size.to_bytes(8, "big"))

        actual_size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                actual_size += len(chunk)

        if actual_size != expected_size:
            raise ValueError(
                f"Fixture file changed while being inspected: "
                f"{relative}"
            )

        total_byte_count += actual_size

    return FixtureSnapshot(
        fixture_id = fixture_id,
        revision = digest.hexdigest(),
        file_count = len(files),
        byte_count = total_byte_count,
    )

def verify_task_fixture(
        task: TaskSpec,
        root: Path,
) -> FixtureSnapshot:
    """Verify that a fixture still matches the task's recorded revision. """

    snapshot = inspect_fixture(
        task.fixture_id,
        root,
    )

    if snapshot.revision != task.fixture_revision:
        raise ValueError(
            f"Fixture revision mismatch for task {task.task_id!r}: "
            f"expected {task.fixture_revision}, "
            f"got {snapshot.revision}"
        )

    return snapshot

