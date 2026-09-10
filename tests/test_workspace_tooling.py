import json
from pathlib import Path
from typing import Any

import pytest

from capability_capsule.runtime.workspace import ReadOnlyWorkspace


def test_workspace_tool_definitions_expose_only_read_and_search() -> None:
    module = __import__(
        "capability_capsule.runtime.tooling", fromlist=["workspace_tool_definitions"]
    )

    definitions = module.workspace_tool_definitions()

    names = [definition["function"]["name"] for definition in definitions]
    assert names == ["read_file", "search_text"]
    assert all(definition["type"] == "function" for definition in definitions)
    assert all("parameters" in definition["function"] for definition in definitions)


def test_execute_read_file_returns_json_result(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.tooling", fromlist=["execute_workspace_tool"])
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    workspace = ReadOnlyWorkspace(tmp_path)

    content = module.execute_workspace_tool(
        workspace,
        "read_file",
        {"relative_path": "app.py", "max_chars": 6},
    )

    result = json.loads(content)
    assert result["relative_path"] == "app.py"
    assert result["text"] == "print("
    assert result["truncated"] is True


def test_execute_search_text_returns_json_matches(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.tooling", fromlist=["execute_workspace_tool"])
    (tmp_path / "app.py").write_text("capsule one\nother\ncapsule two\n", encoding="utf-8")
    workspace = ReadOnlyWorkspace(tmp_path)

    content = module.execute_workspace_tool(
        workspace,
        "search_text",
        {"query": "capsule", "max_results": 1},
    )

    result = json.loads(content)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["relative_path"] == "app.py"
    assert result["matches"][0]["line_number"] == 1


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("delete_file", {}),
        ("read_file", {}),
        ("read_file", {"relative_path": "app.py", "unknown": True}),
        ("read_file", {"relative_path": "app.py", "max_chars": 100_000}),
        ("search_text", {"query": "x", "max_results": 1_000}),
    ],
)
def test_execute_workspace_tool_rejects_unknown_or_unbounded_requests(
    tmp_path: Path,
    name: str,
    arguments: dict[str, Any],
) -> None:
    module = __import__("capability_capsule.runtime.tooling", fromlist=["execute_workspace_tool"])
    workspace = ReadOnlyWorkspace(tmp_path)

    with pytest.raises(ValueError):
        module.execute_workspace_tool(workspace, name, arguments)
