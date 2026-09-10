"""Safe read_only tools for a local agent workspace."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.scanner.documents import read_repository_documents
from capability_capsule.scanner.repo import scan_repository


class FileReadResult(BaseModel):
    """Bounded text returned from one workspace file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    text: str
    truncated: bool


class TextSearchMatch(BaseModel):
    """One line containing a workspace text match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    line_number: int = Field(gt=0)
    line_text: str


class ReadOnlyWorkspace:
    """Expose bounded file reading and literal text search inside one root"""

    def __init__(
        self,
        root: Path,
        *,
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        repository_root = root.resolve(strict=True)

        if not repository_root.is_dir():
            raise NotADirectoryError(repository_root)

        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")

        self._root = repository_root
        self._max_file_size_bytes = max_file_size_bytes

    @property
    def root(self) -> Path:
        """Return the validated workspace root"""
        return self._root

    def _validate_relative_path(self, relative_path: str) -> str:
        if not relative_path.strip():
            raise ValueError("relative_path must not be blank")

        path = Path(relative_path)

        if path.is_absolute():
            raise ValueError("Workspace paths must be relative")

        candidate = (self._root / path).resolve(strict=False)

        try:
            candidate.relative_to(self._root)
        except ValueError as error:
            raise ValueError("Path is outside the workspace") from error

        return path.as_posix()

    def _documents(self) -> dict[str, str]:
        files = scan_repository(self._root)
        documents = read_repository_documents(
            self._root,
            files,
            max_file_size_bytes=self._max_file_size_bytes,
        )

        return {document.relative_path: document.text for document in documents}

    def read_file(self, relative_path: str, *, max_chars: int = 12_000) -> FileReadResult:
        """Read bounded UTF-8 text from one visible workspace file."""

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")

        normalized_path = self._validate_relative_path(relative_path)
        documents = self._documents()

        try:
            text = documents[normalized_path]

        except KeyError as error:
            raise FileNotFoundError(normalized_path) from error

        return FileReadResult(
            relative_path=normalized_path,
            text=text[:max_chars],
            truncated=len(text) > max_chars,
        )

    def search_text(
        self,
        query: str,
        *,
        max_results: int = 20,
    ) -> tuple[TextSearchMatch, ...]:
        """Search visible UTF-8 files for case_insensitive literal text."""

        if not query.strip():
            raise ValueError("query must not be blank")

        if max_results <= 0:
            raise ValueError("max_results must be positive")

        needle = query.casefold()
        matches: list[TextSearchMatch] = []

        for relative_path, text in self._documents().items():
            for line_number, line_text in enumerate(text.splitlines(), start=1):
                if needle not in line_text.casefold():
                    continue

                matches.append(
                    TextSearchMatch(
                        relative_path=relative_path,
                        line_number=line_number,
                        line_text=line_text,
                    )
                )

                if len(matches) >= max_results:
                    return tuple(matches)

        return tuple(matches)
