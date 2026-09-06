from pathlib import Path

import pytest

from capability_capsule.scanner.repo import scan_repository


def test_scan_repository_is_sorted_and_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / ".git").mkdir()

    (tmp_path / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("capsule", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "build" / "output.bin").write_bytes(b"ignored")
    (tmp_path / "debug.log").write_text("ignored", encoding="utf-8")
    (tmp_path / ".git" / "HEAD").write_text("ignored", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert [item.relative_path for item in files] == [
        ".gitignore",
        "README.md",
        "src/main.py",
    ]
    assert files[1].size_bytes == len("capsule")


def test_scan_repository_rejects_a_file_as_root(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_repository(file_path)
