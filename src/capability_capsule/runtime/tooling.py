"""Ollama tool definitions and dispatch for a read-only workspace."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.runtime.workspace import (
    ReadOnlyWorkspace,
    TextSearchMatch,
)


class ReadFileArguments(BaseModel):
    """Validated arguments for the read_file tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    max_chars: int = Field(default=12_000, gt=0, le=50_000)


class SearchTextArguments(BaseModel):
    """Validated arguments for the search_text tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    max_results: int = Field(default=20, gt=0, le=100)


class SearchTextResult(BaseModel):
    """JSON-serializable collection of workspace search matches."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[TextSearchMatch, ...]


def workspace_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return the read-only tools in Ollama function-calling format."""

    return (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read bounded UTF-8 text from a repository-relative workspace file."
                ),
                "parameters": ReadFileArguments.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": (
                    "Search visible workspace text files for a case-insensitive literal string."
                ),
                "parameters": SearchTextArguments.model_json_schema(),
            },
        },
    )


def execute_workspace_tool(
    workspace: ReadOnlyWorkspace,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Validate and execute one allowed workspace tool."""

    if name == "read_file":
        read_arguments = ReadFileArguments.model_validate(arguments)
        result = workspace.read_file(
            read_arguments.relative_path,
            max_chars=read_arguments.max_chars,
        )
        return result.model_dump_json()

    if name == "search_text":
        search_arguments = SearchTextArguments.model_validate(arguments)
        matches = workspace.search_text(
            search_arguments.query,
            max_results=search_arguments.max_results,
        )
        return SearchTextResult(matches=matches).model_dump_json()

    raise ValueError(f"Unknown workspace tool: {name}")
