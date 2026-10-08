#!/usr/bin/env python3
"""Create a privacy-filtered hardware profile for Student-aware data collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ALLOWED_PURPOSES = (
    "training",
    "inference",
    "evaluation",
    "quantization",
    "benchmarking",
    "data_generation",
)

def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _linux_cpu_model() -> str | None:
    text = _read_text(Path("/proc/cpuinfo"))
    if text:
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                return value.strip() or None
    return platform.processor().strip() or None


def _linux_memory() -> tuple[int | None, int | None]:
    text = _read_text(Path("/proc/meminfo"))
    if not text:
        return None, None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        fields = value.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def _windows_snapshot() -> dict[str, Any] | None:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        return None
    script = (
        "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1 Name,"
        "NumberOfCores,NumberOfLogicalProcessors;"
        "$os=Get-CimInstance Win32_OperatingSystem|Select-Object TotalVisibleMemorySize,"
        "FreePhysicalMemory;"
        "$gpu=@(Get-CimInstance Win32_VideoController|Select-Object Name,AdapterRAM);"
        "[pscustomobject]@{cpu=$cpu;os=$os;gpu=$gpu}|ConvertTo-Json -Depth 4 -Compress"
    )
    output = _run([executable, "-NoProfile", "-NonInteractive", "-Command", script])
    if output is None:
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _mac_value(name: str) -> str | None:
    executable = shutil.which("sysctl")
    return _run([executable, "-n", name]) if executable else None


def _detect_execution_environment() -> str:
    release = platform.release().lower()
    if os.environ.get("WSL_INTEROP") or "microsoft" in release:
        return "wsl"
    if Path("/.dockerenv").exists():
        return "container"
    return "native"


def _nvidia_accelerators() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    output = _run(
        [
            executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    accelerators: list[dict[str, Any]] = []
    for line in output.splitlines():
        name, separator, memory_mib = line.rpartition(",")
        if separator and memory_mib.strip().isdigit():
            memory_bytes = (
                int(memory_mib.strip()) * 1024 * 1024
            )
        else:
            memory_bytes = None
        accelerators.append(
            {
                "kind": "gpu",
                "vendor": "nvidia",
                "name": name.strip() if separator else line.strip(),
                "reported_memory_bytes": memory_bytes,
                "memory_semantics": "dedicated",
                "compute_backends": ["cuda"],
            }
        )
    return accelerators


def _windows_accelerators(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    raw = snapshot.get("gpu", [])
    devices = raw if isinstance(raw, list) else [raw]
    accelerators: list[dict[str, Any]] = []
    for device in devices:
        if not isinstance(device, dict) or not device.get("Name"):
            continue
        name = str(device["Name"])
        lowered = name.lower()
        vendor = "amd" if "amd" in lowered or "radeon" in lowered else None
        if "nvidia" in lowered:
            vendor = "nvidia"
        elif "intel" in lowered:
            vendor = "intel"
        accelerators.append(
            {
                "kind": "gpu",
                "vendor": vendor,
                "name": name,
                "reported_memory_bytes": device.get("AdapterRAM"),
                "memory_semantics": "windows_adapter_report; may exclude shared system memory",
                "compute_backends": [],
            }
        )
    return accelerators

def _normalize_purposes(purposes: Sequence[str],) -> tuple[str, ...]:
    unknown = sorted(
        set(purposes) - set(ALLOWED_PURPOSES)
    )

    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(
            f"Unknown environment purpose: {joined}"
        )

    return tuple(dict.fromkeys(purposes))


def _normalize_purposes(
    purposes: Sequence[str],
) -> tuple[str, ...]:
    unknown = sorted(
        set(purposes) - set(ALLOWED_PURPOSES)
    )

    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(
            f"Unknown environment purposes: {joined}"
        )

    return tuple(dict.fromkeys(purposes))


def inspect_hardware(
    purposes: Sequence[str],
    verification_status: str,
) -> dict[str, Any]:
    normalized_purposes = _normalize_purposes(purposes)

    if verification_status not in {
        "unverified",
        "confirmed",
    }:
        raise ValueError(
            "verification_status must be "
            "'unverified' or 'confirmed'"
        )

    if (
        verification_status == "confirmed"
        and not normalized_purposes
    ):
        raise ValueError(
            "A confirmed environment requires "
            "at least one purpose"
        )

    system = platform.system().lower()
    execution_environment = (
        _detect_execution_environment()
    )

    warnings: list[str] = []
    total_memory: int | None = None
    available_memory: int | None = None
    cpu_model: str | None = (
        platform.processor().strip() or None
    )
    physical_cores: int | None = None
    logical_cores = os.cpu_count()
    accelerators = _nvidia_accelerators()

    windows = (
        _windows_snapshot()
        if system == "windows"
        else None
    )

    if windows:
        cpu = windows.get("cpu", {})
        operating_system = windows.get("os", {})

        if isinstance(cpu, dict):
            cpu_model = cpu.get("Name") or cpu_model
            physical_cores = cpu.get("NumberOfCores")
            logical_cores = (
                cpu.get("NumberOfLogicalProcessors")
                or logical_cores
            )

        if isinstance(operating_system, dict):
            total_kib = operating_system.get(
                "TotalVisibleMemorySize"
            )
            free_kib = operating_system.get(
                "FreePhysicalMemory"
            )

            total_memory = (
                int(total_kib) * 1024
                if total_kib is not None
                else None
            )
            available_memory = (
                int(free_kib) * 1024
                if free_kib is not None
                else None
            )

        if not accelerators:
            accelerators = _windows_accelerators(
                windows
            )

    elif system == "linux":
        cpu_model = _linux_cpu_model() or cpu_model
        (
            total_memory,
            available_memory,
        ) = _linux_memory()

    elif system == "darwin":
        cpu_model = (
            _mac_value("machdep.cpu.brand_string")
            or _mac_value("hw.model")
        )

        memory = _mac_value("hw.memsize")
        physical = _mac_value("hw.physicalcpu")

        total_memory = (
            int(memory)
            if memory and memory.isdigit()
            else None
        )
        physical_cores = (
            int(physical)
            if physical and physical.isdigit()
            else None
        )

    if cpu_model is None:
        warnings.append(
            "CPU model could not be detected."
        )

    if total_memory is None:
        warnings.append(
            "Effective memory limit could not be detected."
        )

    if not accelerators:
        warnings.append(
            "No supported GPU detector reported "
            "an accelerator."
        )

    if verification_status == "unverified":
        warnings.append(
            "The observed tool-execution host has not "
            "been confirmed for the declared purposes."
        )

    hardware_identity = {
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "execution_environment": (
                execution_environment
            ),
        },
        "cpu": {
            "model": cpu_model,
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
        },
        "memory": {
            "effective_total_bytes": total_memory,
        },
        "accelerators": accelerators,
    }

    canonical_identity = json.dumps(
        hardware_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical_identity.encode("utf-8")
    ).hexdigest()

    profile_id = f"hardware-{digest[:16]}"

    return {
        "schema_version": "0.2",
        "profile_id": profile_id,
        "captured_at": datetime.now(UTC).isoformat(),
        "observation_scope": "tool_execution_host",
        "model_inference_location": "not_inferred",
        "environment": {
            "verification_status": (
                verification_status
            ),
            "purposes": list(normalized_purposes),
        },
        **hardware_identity,
        "memory_observation": {
            "available_bytes_at_capture": (
                available_memory
            ),
        },
        "privacy": {
            "excluded": [
                "hostname",
                "username",
                "serial_numbers",
                "network_addresses",
                "environment_variables",
            ],
        },
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this assignment-approved path.")
    parser.add_argument("--verification-status", choices=("unverified", "confirmed"), default="unverified", help=("Whether the declared environment purposes have been confirmed."))
    parser.add_argument("--purpose", action="append", choices=ALLOWED_PURPOSES, default=[], help=("Purpose surved by this environment. Repeat the option for multiple purpose."))
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = inspect_hardware(
        purposes = args.purpose,
        verification_status=(
            args.verification_status
        ),
    )

    payload = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
