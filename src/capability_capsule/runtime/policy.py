"""Operation-specific authorization policy for agent tools"""

from collections.abc import Collection
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PolicyLevel(StrEnum):
    """Authorization level assigned to one tool operation."""

    ALLOW = "allow"
    APPROVAL = "approval"
    EXPLICIT = "explicit"
    DENY = "deny"


class ToolPolicyDecision(BaseModel):
    """Policy classification for one requested tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    level: PolicyLevel
    reason: str = Field(min_length=1)


_TOOL_RULES: dict[str, tuple[PolicyLevel, str]] = {
    "read_file": (PolicyLevel.ALLOW, "Bounded read inside the validated workspace"),
    "search_text": (PolicyLevel.ALLOW, "Bounded search inside the validated workspace"),
    "git_status": (PolicyLevel.ALLOW, "read_only Git working-tree inspection."),
    "git_diff": (PolicyLevel.ALLOW, "Read-only Git change inspection."),
    "git_log": (PolicyLevel.ALLOW, "Read-only Git history inspection."),
    "apply_patch": (PolicyLevel.APPROVAL, "Modifies workspace files and requires user approval."),
    "run_tests": (PolicyLevel.APPROVAL, "Executes repository code and requires user approval."),
    "run_checks": (PolicyLevel.APPROVAL, "Executes project tooling and requires user approval."),
    "git_add": (PolicyLevel.APPROVAL, "Changes the Git index and requires user approval."),
    "git_commit": (PolicyLevel.APPROVAL, "Creates local Git history and may execute hooks."),
    "git_push": (PolicyLevel.EXPLICIT, "Changes a remote repository and requires fresh approval."),
    "sync_dependencies": (PolicyLevel.EXPLICIT, "May install packages or access the network"),
    "network_request": (PolicyLevel.EXPLICIT, "Accesses an external network destination."),
    "git_reset_hard": (PolicyLevel.DENY, "Can irreversibly discard workspace changes."),
    "git_clean": (PolicyLevel.DENY, "Can irreversibly delete untracked workspace files."),
    "git_force_push": (PolicyLevel.DENY, "Can overwrite remote repository history."),
    "git_config_global": (PolicyLevel.DENY, "Would modify configuration outside the workspace."),
}


def classify_tool(tool_name: str) -> ToolPolicyDecision:
    """Classify a tool using an exact operation name"""

    if not tool_name.strip():
        raise ValueError("tool_name must not be blank")

    if tool_name != tool_name.strip():
        raise ValueError("tool_name must not contain surrounding whitespace")

    rule = _TOOL_RULES.get(tool_name)

    if rule is None:
        return ToolPolicyDecision(
            tool_name=tool_name,
            level=PolicyLevel.DENY,
            reason="Unknown tools are denied by default.",
        )

    level, reason = rule

    return ToolPolicyDecision(
        tool_name=tool_name,
        level=level,
        reason=reason,
    )


def is_tool_authorized(
    decision: ToolPolicyDecision,
    *,
    task_approvals: Collection[str] = frozenset(),
    invocation_approved: bool = False,
) -> bool:
    """Return whether the available approval grants authorize a tool"""

    if decision.level is PolicyLevel.ALLOW:
        return True

    if decision.level is PolicyLevel.DENY:
        return False

    if decision.level is PolicyLevel.EXPLICIT:
        return invocation_approved

    return invocation_approved or decision.tool_name in task_approvals
