"""Load host-controlled tool policy overrides from TOML"""

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from capability_capsule.runtime.policy import (
    PolicyLevel,
    ToolPolicyDecision,
    classify_tool,
)

_MAX_POLICY_BYTES = 1_000_000

_POLICY_ORDER = {
    PolicyLevel.ALLOW: 0,
    PolicyLevel.APPROVAL: 1,
    PolicyLevel.EXPLICIT: 2,
    PolicyLevel.DENY: 3,
}


class ToolPolicyConfig(BaseModel):
    """Validated host policy that can only tighten built-in rules"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    tools: dict[str, PolicyLevel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_overrides_only_tighten(self) -> Self:
        """Reject overrides that weaken a built-in policy level"""

        for tool_name, override_level in self.tools.items():
            baseline = classify_tool(tool_name)

            if _POLICY_ORDER[override_level] < _POLICY_ORDER[baseline.level]:
                raise ValueError(
                    f"Policy override for {tool_name} would weaken "
                    f"{baseline.level.value} to {override_level.value}"
                )

        return self

    def decision(self, tool_name: str) -> ToolPolicyDecision:
        """Return the built-in decision with an optional stricter override"""

        baseline = classify_tool(tool_name)
        override_level = self.tools.get(tool_name)
        if override_level is None or override_level is baseline.level:
            return baseline

        return ToolPolicyDecision(
            tool_name=tool_name,
            level=override_level,
            reason=(
                f"Host policy tightened the built-in "
                f"{baseline.level.value} rule to {override_level.value}."
            ),
        )


def load_tool_policy(path: Path) -> ToolPolicyConfig:
    """Read and validate a non-empty TOML policy file."""

    if path.is_symlink():
        raise ValueError("Policy path must not be a symbolic link")

    source = path.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    if source.stat().st_size > _MAX_POLICY_BYTES:
        raise ValueError("Policy file exceeds the size limit")

    with source.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    return ToolPolicyConfig.model_validate(data)
