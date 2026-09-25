from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_probe() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / ".agents"
        / "skills"
        / "capsule-teacher"
        / "scripts"
        / "inspect_hardware.py"
    )
    spec = importlib.util.spec_from_file_location("capsule_teacher_hardware_probe", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_excludes_identity_data_and_marks_observation_scope() -> None:
    profile = load_probe().inspect_hardware(
        purposes=("inference", "evaluation"),
        verification_status="confirmed",
    )

    assert profile["schema_version"] == "0.2"
    assert profile["profile_id"].startswith("hardware-")
    assert profile["observation_scope"] == "tool_execution_host"
    assert profile["model_inference_location"] == "not_inferred"
    assert profile["environment"] == {
        "verification_status": "confirmed",
        "purposes": ["inference", "evaluation"],
    }
    assert "hostname" in profile["privacy"]["excluded"]
    assert not {"hostname", "username", "serial_number", "ip_address"} & profile.keys()


def test_environment_metadata_does_not_change_hardware_identity() -> None:
    probe = load_probe()

    unverified = probe.inspect_hardware(
        purposes=(),
        verification_status="unverified",
    )
    confirmed = probe.inspect_hardware(
        purposes=("inference", "benchmarking"),
        verification_status="confirmed",
    )

    assert unverified["profile_id"] == confirmed["profile_id"]
    assert unverified["environment"]["verification_status"] == "unverified"
    assert confirmed["environment"]["purposes"] == ["inference", "benchmarking"]


def test_confirmed_environment_requires_at_least_one_purpose() -> None:
    probe = load_probe()

    with pytest.raises(ValueError, match="at least one purpose"):
        probe.inspect_hardware(
            purposes=(),
            verification_status="confirmed",
        )


def test_probe_normalizes_duplicate_purposes_without_changing_order() -> None:
    profile = load_probe().inspect_hardware(
        purposes=("evaluation", "inference", "evaluation"),
        verification_status="confirmed",
    )

    assert profile["environment"]["purposes"] == ["evaluation", "inference"]
