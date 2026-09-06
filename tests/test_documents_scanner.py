from pathlib import Path

import pytest

from capability_capsule.manifest import SourceType
from capability_capsule.scanner.documents import read_repository_documents
from capability_capsule.scanner.repo import RepositoryFile


def test_read_repository_documents_filters_and_sorts_text_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "assets").mkdir()

    python_file = tmp_path / "src" / "main.py"
    markdown_file = tmp_path / "notes" / "design.md"
    image_file = tmp_path / "assets" / "logo.png"
    large_file = tmp_path / "notes" / "large.txt"
    invalid_utf8_file = tmp_path / "notes" / "invalid.txt"

    python_file.write_text("print('ok')\n", encoding="utf-8")
    markdown_file.write_text("# Design\n", encoding="utf-8")
    image_file.write_bytes(b"\x89PNG\r\n")
    large_file.write_text("x" * 64, encoding="utf-8")
    invalid_utf8_file.write_bytes(b"\xff\xfe")

    files = (
        _record(tmp_path, python_file),
        _record(tmp_path, invalid_utf8_file),
        _record(tmp_path, image_file),
        _record(tmp_path, markdown_file),
        _record(tmp_path, large_file),
    )

    documents = read_repository_documents(
        tmp_path,
        files,
        max_file_size_bytes=32,
    )

    assert [document.relative_path for document in documents] == [
        "notes/design.md",
        "src/main.py",
    ]
    assert documents[0].text == "# Design\n"
    assert documents[0].source_type is SourceType.REPO
    assert documents[1].size_bytes == len("print('ok')\n")


def test_read_repository_documents_rejects_path_outside_root(tmp_path: Path) -> None:
    outside_file = tmp_path.parent / "outside.py"
    record = RepositoryFile(
        relative_path="../outside.py",
        size_bytes=0,
    )

    with pytest.raises(ValueError):
        read_repository_documents(tmp_path, (record,))

    assert not outside_file.exists()


def test_read_repository_documents_requires_positive_size_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_repository_documents(tmp_path, (), max_file_size_bytes=0)


def _record(root: Path, path: Path) -> RepositoryFile:
    return RepositoryFile(
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
    )
