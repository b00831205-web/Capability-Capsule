import pytest


@pytest.mark.parametrize(
    ("tool_name", "expected_level"),
    [
        ("read_file", "allow"),
        ("search_text", "allow"),
        ("git_status", "allow"),
        ("git_diff", "allow"),
        ("git_log", "allow"),
        ("apply_patch", "approval"),
        ("run_tests", "approval"),
        ("run_checks", "approval"),
        ("git_add", "approval"),
        ("git_commit", "approval"),
        ("git_push", "explicit"),
        ("sync_dependencies", "explicit"),
        ("network_request", "explicit"),
        ("git_reset_hard", "deny"),
        ("git_clean", "deny"),
        ("git_force_push", "deny"),
        ("git_config_global", "deny"),
        ("unknown_tool", "deny"),
    ],
)
def test_classify_tool_uses_operation_specific_policy(tool_name: str, expected_level: str) -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["classify_tool"])

    decision = module.classify_tool(tool_name)

    assert decision.tool_name == tool_name
    assert decision.level.value == expected_level
    assert bool(decision.reason.strip())


def test_unknown_and_blank_tools_are_not_silently_allowed() -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["classify_tool"])

    assert module.classify_tool("something_new").level.value == "deny"

    with pytest.raises(ValueError):
        module.classify_tool("")


def test_auto_tools_execute_without_approval() -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["is_tool_authorized"])
    decision = module.classify_tool("read_file")

    assert module.is_tool_authorized(decision) is True


def test_approval_tools_accept_once_or_task_approval() -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["is_tool_authorized"])
    decision = module.classify_tool("git_commit")

    assert module.is_tool_authorized(decision) is False
    assert module.is_tool_authorized(decision, invocation_approved=True) is True
    assert (
        module.is_tool_authorized(
            decision,
            task_approvals=frozenset({"git_commit"}),
        )
        is True
    )


def test_explicit_tools_require_fresh_invocation_approval() -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["is_tool_authorized"])
    decision = module.classify_tool("git_push")

    assert module.is_tool_authorized(decision) is False
    assert (
        module.is_tool_authorized(
            decision,
            task_approvals=frozenset({"git_push"}),
        )
        is False
    )
    assert module.is_tool_authorized(decision, invocation_approved=True) is True


def test_denied_tools_cannot_be_approved() -> None:
    module = __import__("capability_capsule.runtime.policy", fromlist=["is_tool_authorized"])
    decision = module.classify_tool("git_reset_hard")

    assert module.is_tool_authorized(decision) is False
    assert module.is_tool_authorized(decision, invocation_approved=True) is False
    assert (
        module.is_tool_authorized(
            decision,
            task_approvals=frozenset({"git_reset_hard"}),
            invocation_approved=True,
        )
        is False
    )
