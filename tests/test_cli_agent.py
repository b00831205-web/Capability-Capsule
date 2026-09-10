import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.runtime.agent import ReadOnlyAgentResult
from capability_capsule.runtime.policy_config import ToolPolicyConfig


@pytest.mark.parametrize("json_output", [False, True])
def test_agent_command_runs_with_workspace_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text('schema_version = "0.1"\n', encoding="utf-8")
    policy = ToolPolicyConfig()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    monkeypatch.setattr(cli, "load_tool_policy", lambda path: policy)

    def run_agent(*args: Any, **kwargs: Any) -> ReadOnlyAgentResult:
        calls.append((args, kwargs))
        return ReadOnlyAgentResult(
            answer="The entry is in app.py.",
            generation_model="qwen3.5:4b",
            tool_call_count=2,
            tool_names=("search_text", "read_file"),
        )

    monkeypatch.setattr(cli, "run_read_only_agent", run_agent)
    arguments = [
        "agent",
        "Find the entry point",
        "--workspace",
        str(workspace),
        "--policy",
        str(policy_path),
        "--max-tool-rounds",
        "3",
        "--max-tool-calls",
        "5",
    ]
    if json_output:
        arguments.append("--json")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert calls[0][0][0] == "Find the entry point"
    assert calls[0][0][1] == workspace
    assert calls[0][1]["policy"] is policy
    assert calls[0][1]["max_tool_rounds"] == 3
    assert calls[0][1]["max_tool_calls"] == 5
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["answer"] == "The entry is in app.py."
        assert payload["tool_call_count"] == 2
    else:
        assert "The entry is in app.py." in result.stdout
        assert "search_text" in result.stdout
        assert "read_file" in result.stdout


def test_agent_command_runs_without_policy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def run_agent(*args: Any, **kwargs: Any) -> ReadOnlyAgentResult:
        calls.append(kwargs)
        return ReadOnlyAgentResult(
            answer="answer",
            generation_model="qwen3.5:4b",
            tool_call_count=0,
            tool_names=(),
        )

    monkeypatch.setattr(cli, "run_read_only_agent", run_agent)
    result = CliRunner().invoke(
        cli.app,
        ["agent", "Explain", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert calls[0]["policy"] is None


def test_agent_command_reports_runtime_or_policy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> ReadOnlyAgentResult:
        raise PermissionError("tool requires approval")

    monkeypatch.setattr(cli, "run_read_only_agent", fail)
    result = CliRunner().invoke(
        cli.app,
        ["agent", "Read a file", "--workspace", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "tool requires approval" in result.output
