import json
from pathlib import Path

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.runtime.policy_config import ToolPolicyConfig


def _tool_response(name: str, arguments: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "done": True,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    }
                ],
            },
        },
    )


def test_agent_executes_read_only_tool_then_answers(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])
    (tmp_path / "app.py").write_text(
        "def capsule_entry():\n    return 'ready'\n",
        encoding="utf-8",
    )
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 128
        if len(requests) == 1:
            tool_names = [tool["function"]["name"] for tool in payload["tools"]]
            assert tool_names == ["read_file", "search_text"]
            return _tool_response("search_text", {"query": "capsule_entry"})

        messages = payload["messages"]
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_name"] == "search_text"
        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["matches"][0]["relative_path"] == "app.py"
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "The entry is in app.py.",
                },
            },
        )

    settings = Settings()
    result = module.run_read_only_agent(
        "Find the capsule entry point",
        tmp_path,
        settings,
        transport=httpx.MockTransport(handle),
    )

    assert result.answer == "The entry is in app.py."
    assert result.generation_model == settings.ollama.generation_model
    assert result.tool_call_count == 1
    assert result.tool_names == ("search_text",)
    assert len(requests) == 2


def test_agent_honors_host_policy_that_requires_approval(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])
    (tmp_path / "app.py").write_text("content", encoding="utf-8")
    policy = ToolPolicyConfig.model_validate({"tools": {"read_file": "approval"}})

    def handle(request: httpx.Request) -> httpx.Response:
        return _tool_response("read_file", {"relative_path": "app.py"})

    with pytest.raises(PermissionError):
        module.run_read_only_agent(
            "Read app.py",
            tmp_path,
            Settings(),
            policy=policy,
            transport=httpx.MockTransport(handle),
        )


def test_agent_rejects_invalid_or_empty_final_response(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"done": True, "message": {"role": "assistant", "content": ""}},
        )

    with pytest.raises(ValueError):
        module.run_read_only_agent(
            "Explain the project",
            tmp_path,
            Settings(),
            transport=httpx.MockTransport(handle),
        )


def test_agent_stops_repeated_tool_calls_at_round_limit(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])
    (tmp_path / "app.py").write_text("content", encoding="utf-8")
    request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return _tool_response("read_file", {"relative_path": "app.py"})

    with pytest.raises(RuntimeError):
        module.run_read_only_agent(
            "Keep reading",
            tmp_path,
            Settings(),
            max_tool_rounds=1,
            transport=httpx.MockTransport(handle),
        )

    assert request_count == 2


def test_agent_stops_batch_tool_calls_at_total_limit(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])
    (tmp_path / "app.py").write_text("content", encoding="utf-8")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"relative_path": "app.py"},
                            }
                        },
                        {
                            "function": {
                                "name": "search_text",
                                "arguments": {"query": "content"},
                            }
                        },
                    ],
                },
            },
        )

    with pytest.raises(RuntimeError):
        module.run_read_only_agent(
            "Inspect the file",
            tmp_path,
            Settings(),
            max_tool_calls=1,
            transport=httpx.MockTransport(handle),
        )


@pytest.mark.parametrize(("task", "max_rounds"), [("", 2), ("task", 0)])
def test_agent_validates_runtime_limits(
    tmp_path: Path,
    task: str,
    max_rounds: int,
) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])

    with pytest.raises(ValueError):
        module.run_read_only_agent(
            task,
            tmp_path,
            Settings(),
            max_tool_rounds=max_rounds,
        )


def test_agent_rejects_non_positive_tool_call_limit(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.agent", fromlist=["run_read_only_agent"])

    with pytest.raises(ValueError):
        module.run_read_only_agent(
            "task",
            tmp_path,
            Settings(),
            max_tool_calls=0,
        )
