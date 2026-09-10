from pathlib import Path

import pytest


def test_load_policy_tightens_selected_tool_levels(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.policy_config", fromlist=["load_tool_policy"])
    path = tmp_path / "policy.toml"
    path.write_text(
        """
schema_version = "0.1"

[tools]
read_file = "approval"
git_commit = "explicit"
network_request = "deny"
""",
        encoding="utf-8",
    )

    policy = module.load_tool_policy(path)

    assert policy.decision("read_file").level.value == "approval"
    assert policy.decision("git_commit").level.value == "explicit"
    assert policy.decision("network_request").level.value == "deny"
    assert policy.decision("search_text").level.value == "allow"


@pytest.mark.parametrize(
    ("tool_name", "level"),
    [
        ("git_commit", "allow"),
        ("git_push", "approval"),
        ("git_reset_hard", "allow"),
        ("unknown_tool", "allow"),
    ],
)
def test_policy_file_cannot_relax_builtin_safety(
    tmp_path: Path,
    tool_name: str,
    level: str,
) -> None:
    module = __import__("capability_capsule.runtime.policy_config", fromlist=["load_tool_policy"])
    path = tmp_path / "policy.toml"
    path.write_text(
        f'schema_version = "0.1"\n[tools]\n{tool_name} = "{level}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.load_tool_policy(path)


def test_policy_rejects_unknown_fields_and_levels(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.policy_config", fromlist=["load_tool_policy"])
    unknown_field = tmp_path / "unknown.toml"
    unknown_field.write_text('schema_version = "0.1"\nextra = true\n', encoding="utf-8")
    invalid_level = tmp_path / "invalid.toml"
    invalid_level.write_text(
        'schema_version = "0.1"\n[tools]\nread_file = "always"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.load_tool_policy(unknown_field)

    with pytest.raises(ValueError):
        module.load_tool_policy(invalid_level)


def test_empty_policy_uses_builtin_defaults(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.policy_config", fromlist=["load_tool_policy"])
    path = tmp_path / "policy.toml"
    path.write_text('schema_version = "0.1"\n', encoding="utf-8")

    policy = module.load_tool_policy(path)

    assert policy.decision("read_file").level.value == "allow"
    assert policy.decision("git_push").level.value == "explicit"
    assert policy.decision("git_clean").level.value == "deny"


def test_policy_rejects_missing_directory_and_symbolic_link(tmp_path: Path) -> None:
    module = __import__("capability_capsule.runtime.policy_config", fromlist=["load_tool_policy"])

    with pytest.raises((OSError, ValueError)):
        module.load_tool_policy(tmp_path / "missing.toml")

    with pytest.raises((OSError, ValueError)):
        module.load_tool_policy(tmp_path)

    target = tmp_path / "target.toml"
    target.write_text('schema_version = "0.1"\n', encoding="utf-8")
    link = tmp_path / "link.toml"
    link.symlink_to(target)

    with pytest.raises((OSError, ValueError)):
        module.load_tool_policy(link)
