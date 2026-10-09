"""Verify immutable run inputs while accounting for explicitly recorded later doc updates."""

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-011-mix006007-28step-train-fit-diagnostic-v1/planning-update.json"
DOCS = frozenset({"MVP_ROADMAP.md", "PENDING_CONFIGURATION_OPTIMIZATIONS.md"})


def input_matches(path, pinned, current, *, update_path=UPDATE):
    if current == pinned:
        return True
    if path not in DOCS or not update_path.exists():
        return False
    update = json.loads(update_path.read_text("utf-8"))
    return (update.get("explicitly_authorized") is True
            and update.get("before", {}).get(path) == pinned
            and update.get("after", {}).get(path) == current
            and sha256((ROOT / path).read_bytes()).hexdigest() == current)
