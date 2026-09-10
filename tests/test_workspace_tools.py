from pathlib import Path

import pytest


def test_read_file_returns_bounded_utf8_text(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.workspace", fromlist=["ReadOnlyWorkspace"])
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("print('capsule')\n", encoding="utf-8")
    workspace = module.ReadOnlyWorkspace(tmp_path)

    result = workspace.read_file("src/app.py", max_chars=8)

    assert result.relative_path == "src/app.py"
    assert result.text == "print('c"
    assert result.truncated is True


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.py", "/tmp/outside.py", "", "   "],
)
def test_read_file_rejects_unsafe_or_blank_paths(tmp_path: Path, relative_path: str) -> None:
    module = __import__("capability_capsule.runtime.workspace", fromlist=["ReadOnlyWorkspace"])
    workspace = module.ReadOnlyWorkspace(tmp_path)

    with pytest.raises((OSError, ValueError)):
        workspace.read_file(relative_path)


def test_read_file_does_not_expose_ignored_files(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.workspace", fromlist=["ReadOnlyWorkspace"])
    (tmp_path / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    workspace = module.ReadOnlyWorkspace(tmp_path)

    with pytest.raises((OSError, ValueError)):
        workspace.read_file("secret.txt")


def test_search_text_returns_case_insensitive_line_matches(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.workspace", fromlist=["ReadOnlyWorkspace"])
    source = tmp_path / "src"
    source.mkdir()
    (source / "first.py").write_text(
        "first line\nCapability Capsule runtime\nlast line\n",
        encoding="utf-8",
    )
    (source / "second.py").write_text("capsule helper\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"capsule\x00binary")
    workspace = module.ReadOnlyWorkspace(tmp_path)

    matches = workspace.search_text("CAPSULE", max_results=10)

    assert [(match.relative_path, match.line_number) for match in matches] == [
        ("src/first.py", 2),
        ("src/second.py", 1),
    ]
    assert all("capsule" in match.line_text.lower() for match in matches)


def test_search_text_respects_result_limit_and_validates_input(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.workspace", fromlist=["ReadOnlyWorkspace"])
    (tmp_path / "notes.txt").write_text("match\nmatch\nmatch\n", encoding="utf-8")
    workspace = module.ReadOnlyWorkspace(tmp_path)

    assert len(workspace.search_text("match", max_results=2)) == 2

    with pytest.raises(ValueError):
        workspace.search_text("")

    with pytest.raises(ValueError):
        workspace.search_text("match", max_results=0)
