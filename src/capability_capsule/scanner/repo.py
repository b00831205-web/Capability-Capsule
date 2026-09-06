"""Deterministic repository file discovery."""

from pathlib import Path

from pathspec import PathSpec
from pydantic import BaseModel, ConfigDict, Field


class RepositoryFile(BaseModel):
    """Metadata for one repository file selected by the scanner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


def scan_repository(root: Path) -> tuple[RepositoryFile, ...]:
    """List repository files in stable path order without reading their contents."""

    repository_root = root.resolve(strict=True)
    if not repository_root.is_dir():
        raise NotADirectoryError(repository_root)

    ignore_spec = _load_ignore_spec(repository_root)
    files: list[RepositoryFile] = []

    for candidate in repository_root.rglob("*"):
        if candidate.is_symlink() or not candidate.is_file():
            continue

        relative_path = candidate.relative_to(repository_root).as_posix()
        if _is_git_metadata(relative_path) or ignore_spec.match_file(relative_path):
            continue

        files.append(
            RepositoryFile(
                relative_path=relative_path,
                size_bytes=candidate.stat().st_size,
            )
        )

    return tuple(sorted(files, key=lambda item: item.relative_path))


def _load_ignore_spec(repository_root: Path) -> PathSpec:
    ignore_file = repository_root / ".gitignore"
    if not ignore_file.is_file():
        return PathSpec.from_lines("gitwildmatch", ())

    return PathSpec.from_lines(
        "gitwildmatch",
        ignore_file.read_text(encoding="utf-8").splitlines(),
    )


def _is_git_metadata(relative_path: str) -> bool:
    return relative_path == ".git" or relative_path.startswith(".git/")
