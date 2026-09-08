"""Safe text-document loading for repository files."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.manifest import SourceType
from capability_capsule.scanner.repo import RepositoryFile

SUPPORTED_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class RepositoryDocument(BaseModel):
    """UTF-8 text loaded from one repository file"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str = Field(min_length=1)
    text: str
    size_bytes: int = Field(ge=0)
    source_type: SourceType = SourceType.REPO


def read_repository_documents(
    root: Path,
    files: Iterable[RepositoryFile],
    *,
    max_file_size_bytes: int = 1_000_000,
) -> tuple[RepositoryDocument, ...]:
    """Load supported UTF-8 text files within a repository root."""

    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive")

    repository_root = root.resolve(strict=True)
    if not repository_root.is_dir():
        raise NotADirectoryError(repository_root)
    documents: list[RepositoryDocument] = []

    for file in files:
        unresolved_path = repository_root / file.relative_path
        candidate = unresolved_path.resolve(strict=False)

        try:
            candidate.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(
                f"file path is outside repository root: {file.relative_path}"
            ) from error

        if unresolved_path.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
            continue

        size_bytes = candidate.stat().st_size
        if size_bytes > max_file_size_bytes:
            continue

        try:
            text = candidate.read_text(encoding="utf-8")

        except UnicodeDecodeError:
            continue

        if "\x00" in text:
            continue

        documents.append(
            RepositoryDocument(relative_path=file.relative_path, text=text, size_bytes=size_bytes)
        )

    return tuple(sorted(documents, key=lambda item: item.relative_path))
